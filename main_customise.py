import argparse
import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    # GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    ForecastReport,
    clean_indents,
)
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
import typeguard

from llm_agent import get_llm_agent_class
from research_agent import ResearchAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress LiteLLM logging
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)
litellm_logger.propagate = False

load_dotenv()

class Q1Bot(ForecastBot):
    """
    This is a customized bot that uses the forecasting-tools library to simplify bot making.

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    However generally the flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research for research_reports_per_question runs
        - Execute respective run_forecast function for `predictions_per_research_report * research_reports_per_question` runs
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```

    Check out https://github.com/Metaculus/forecasting-tools for a full set of features from this package.
    Most notably there is a built in benchmarker that integrates with ForecastBot objects.
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    )
    async def run_research(self, question: MetaculusQuestion) -> str:

        # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
        # await self._rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
        logger.debug(f"Ques: {question}")
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("SERPER_API_KEY"):
                research = await self._call_serper_searcher(
                    serper_api_key=os.getenv("SERPER_API_KEY"),
                    question=question.question_text
                )
            elif os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = AskNewsSearcher(
                    client_id=os.getenv("ASKNEWS_CLIENT_ID"),
                    client_secret=os.getenv("ASKNEWS_SECRET"),
                ).get_formatted_news(question.question_text)
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(question.question_text)
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(question.question_text, use_open_router=True)
            else:
                research = ""
            logger.info(f"Found Research for {question.page_url}:\n{research}")
            return research

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro" # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search.
        # model = GeneralLlm(
        #     model=model_name,
        #     temperature=0.1,
        # )
        # response = await model.invoke(prompt)
        response = ""
        return response

    async def _call_serper_searcher(self, serper_api_key: str, question: str) -> str:
        researchAgent = ResearchAgent(serper_api_key=serper_api_key, breadth=5)
        logger.info(f"Questions: {question}")
        response = await researchAgent.research(question=question)
        logger.info(f"Response: {response}")
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around a search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self._get_final_decision_llm(),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    def _get_final_decision_llm(self):# -> GeneralLlm:
        model = None
        if os.getenv("NIM_API_KEY"):
            model_name = os.getenv("DEFAULT_AI_MODEL")
            model = get_llm_agent_class(model=model_name)(
                model=model_name,
                temperature=0.3
            )
        elif os.getenv("METACULUS_TOKEN"):
            model = get_llm_agent_class(model="claude-3-5-sonnet-20241022")(
                model="claude-3-5-sonnet-20241022",
                temperature=0.3
            )
        # elif os.getenv("OPENAI_API_KEY"):
        #     model = GeneralLlm(model="gpt-4o", temperature=0.3)
        # elif os.getenv("ANTHROPIC_API_KEY"):
        #     model = GeneralLlm(model="claude-3-5-sonnet-20241022", temperature=0.3)
        # elif os.getenv("OPENROUTER_API_KEY"):
        #     model = GeneralLlm(model="openrouter/openai/gpt-4o", temperature=0.3)
        else:
            raise ValueError("No API key for final_decision_llm found")
        return model

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Recall the question you are forecasting:
            {question.question_text}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A few reasons why the answer might be No. Rate the strength of each reason on a scale of 1-10.
            (d) A few reasons why the answer might be Yes. Rate the strength of each reason on a scale of 1-10.
            (e) Output an initial probability as: "Tentative Probability: ZZ%", 0-100.
            (f) Reflect on your answer, performing sanity checks and mentioning any additional knowledge 
                or background information which may be relevant. Check for over/underconfidence,
                improper treatment of conjunctive or disjunctive conditions (only if applicable),
                and other forecasting biases when reviewing your reasoning. Finally, aggregate all of your previous
                reasoning and key factors that inform your final forecast.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self._get_final_decision_llm().async_completions([dict(role="user", content=prompt)])
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A few reasons for different options. Rate the strength of each reason on a scale of 1-10.
            (d) Output an initial probabilities for the N options in this order {question.options} as:
                "Tentative:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                Option_N: Probability_N"
            (e) Reflect on your answer, performing sanity checks and mentioning any additional knowledge 
                or background information which may be relevant. Check for over/underconfidence,
                improper treatment of conjunctive or disjunctive conditions (only if applicable),
                and other forecasting biases when reviewing your reasoning. Finally, aggregate all of your previous
                reasoning and key factors that inform your final forecast.

            You write your rationale remembering that (1) good forecasters put extra
            weight on the status quo outcome since the world changes slowly most of the time,
            and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self._get_final_decision_llm().async_completions([dict(role="user", content=prompt)])
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self._get_final_decision_llm().async_completions([dict(role="user", content=prompt)])
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message

def summarize_reports(forecast_reports: list[ForecastReport | BaseException]) -> None:
    valid_reports = [
        report for report in forecast_reports if isinstance(report, ForecastReport)
    ]
    exceptions = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]
    minor_exceptions = [
        report.errors for report in valid_reports if report.errors
    ]

    for report in valid_reports:
        question_summary = clean_indents(f"""
            URL: {report.question.page_url}
            Errors: {report.errors}
            Summary:
            {report.summary}
            ---------------------------------------------------------
        """)
        logger.info(question_summary)

    if exceptions:
        raise RuntimeError(
            f"{len(exceptions)} errors occurred while forecasting: {exceptions}"
        )
    if minor_exceptions:
        logger.error(
            f"{len(minor_exceptions)} minor exceptions occurred while forecasting: {minor_exceptions}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Q1 forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    customized_bot = Q1Bot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        Q4_2024_AI_BENCHMARKING_ID = 32506
        Q1_2025_AI_BENCHMARKING_ID = 32627
        forecast_reports = asyncio.run(
            customized_bot.forecast_on_tournament(
                Q1_2025_AI_BENCHMARKING_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        Q1_2025_QUARTERLY_CUP_ID = 32630
        customized_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            customized_bot.forecast_on_tournament(
                Q1_2025_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        customized_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            customized_bot.forecast_questions(
                questions, return_exceptions=True
            )
        )
    forecast_reports = typeguard.check_type(forecast_reports, list[ForecastReport | BaseException])
    logger.debug(forecast_reports)
    summarize_reports(forecast_reports)

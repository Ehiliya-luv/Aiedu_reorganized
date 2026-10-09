"""Short result-evaluation entry point.

The implementation is shared with ``evaluate_question_reasoning`` so that
question and reasoning evaluations use exactly the same remote Judge contract.
"""

from .evaluate_question_reasoning import main


if __name__ == "__main__":
    main()

from mild_steel_bars import logger
from mild_steel_bars.config.configuration import ConfigurationManager
from mild_steel_bars.components.tuningAndEvaluation import HyperparameterTuningAndEvaluation

STAGE_NAME = "Model Tuning And Evaluation Stage"

class ModelTuningAndEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        tune_and_eval = config.get_hyperparameter_tuning_and_evaluation_config()
        tuning_and_eval = HyperparameterTuningAndEvaluation(config=tune_and_eval)
        tuning_and_eval.tuningAndEvaluation()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTuningAndEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

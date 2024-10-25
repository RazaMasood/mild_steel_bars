from mild_steel_bars import logger
from mild_steel_bars.config.configuration import ConfigurationManager
from mild_steel_bars.components.trainingAndEvaluation import TrainingAndEvaluation

STAGE_NAME = "Model Training And Evaluation Stage"

class ModelTrainingAndEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_evaluation_config = config.get_model_trainer_and_evaluation_config()
        trainingAndEvl = TrainingAndEvaluation(config=model_training_evaluation_config)
        trainingAndEvl.trainAndEvaluate()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingAndEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e

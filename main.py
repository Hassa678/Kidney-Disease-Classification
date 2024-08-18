from kidney_disease import logger
from kidney_disease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from kidney_disease.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from kidney_disease.pipeline.stage_03_model_training import ModelTrainingPipeline



STAGE_NAME = "Training"


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
from defaults import *
from utils.helpfuns import *
from .models import Model
# from .invaemodule import FinVAEmodule
from self_supervised.BYOL.wrappers import BYOLWrapper

class SupervisedWrapper(BYOLWrapper):
    def __init__(self, parameters):
        super().__init__(parameters)

    def init_model(self):
        # DDP broadcasts model states from rank 0 process to all other processes
        # in the DDP constructor, you don’t need to worry about different DDP processes
        # start from different model parameter initial values.

        # init model and wrap it with DINO
        if hasattr(transformers, self.model_params.backbone_type):
            student_params = deepcopy(self.model_params)
            student_params.transformers_params.update(drop_path_rate=0.1)
            student = Classifier(student_params)
        else:
            student = Classifier(self.model_params)

        if self.transfer_learning_params.use_pretrained:
            pretrained_path = self.transfer_learning_params.pretrained_path
            pretrained_model_name = self.transfer_learning_params.pretrained_model_name
            if not pretrained_path:
                pretrained_path = os.path.join(
                    self.training_params.save_dir, "checkpoints"
                )
            pretrained_path = os.path.join(pretrained_path, pretrained_model_name)

        model = Model(student)
        if self.transfer_learning_params.use_pretrained:
            load_from_pretrained(model, pretrained_path, strict=False, drop_fc=True)

        if ddp_is_on():
            model = DDP(model, device_ids=[self.device_id])

        return model

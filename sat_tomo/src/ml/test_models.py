import ml.models as models
from torchinfo import summary, ModelStatistics
import cfg as config

def size(info):
    return ModelStatistics.to_megabytes(
                info.total_input + info.total_output_bytes + info.total_param_bytes
            )

def run_tests():

    data = []

    cfg = config.Config()

    for m in cfg.train_models:
        model = eval(f"models.{m}(cfg)")


        info = summary(model, input_size=model.input_size, verbose=0)
        print(f"{m}: {info.trainable_params}/{info.total_params} [{size(info):.2f} MB]")
        data.append((m,info.trainable_params,size(info)))
        if "ResNet" in m:
            p0 = info.trainable_params

    for m, p, s in data:
        print(f"{m},{s:.2f},{p},{(p/p0-1.0)*100.0:.4f}")
        
    print("TEST")
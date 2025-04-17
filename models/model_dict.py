from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samus.build_sam_us import samus_model_registry
from models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "AutoSAMUS":
        model = autosamus_model_registry['vit_b'](args=args, checkpoint=opt.load_path)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model

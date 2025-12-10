from _eval_physbench import *


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

    _max_new_tokens = 2048
    _model = lambda x: 'A'
    _model_name = 'all_A'
    _model = Qwen2_5_VL(batch_output=False, max_new_tokens=_max_new_tokens)
    _model_name = 'qwenvl_raw'
    _model = LLaVA_NeXT_Video(max_new_tokens=_max_new_tokens)
    _model_name = 'llavanv_raw'
    _model = InternVL3_5('/home/zhipang/PhysicalDynamics/data/llama_factory_data/data/saves/internvl_cls_20000.merged', max_new_tokens=_max_new_tokens)
    _model_name = 'internvl_cls_20000'
    _model = InternVL3_5(max_new_tokens=_max_new_tokens)
    _model_name = 'internvl_raw'

    eval_physbench(
        model=ModelToBeEvaluated(_model, _model_name),
        model_name=_model_name,
        just_val=False,
    )
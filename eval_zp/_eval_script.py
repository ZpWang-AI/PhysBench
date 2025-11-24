from _eval_physbench import *


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    _model = lambda x: 'A'
    _model_name = 'all_A'
    _model = QwenVL(batch_output=False)
    _model_name = 'qwenvl_raw'
    _model = LLaVA_NeXT_Video()
    _model_name = 'llavanv_raw'
    _model = InternVL3_5()
    _model_name = 'internvl_raw'
    _model = InternVL3_5('/home/zhipang/PhysicalDynamics/data/llama_factory_data/data/saves/internvl_cls_20000.merged')
    _model_name = 'internvl_cls_20000'

    eval_physbench(
        model=ModelToBeEvaluated(_model),
        model_name=_model_name,
        just_val=True,
    )
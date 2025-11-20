from utils_zp import *
from llm_zp import *


SRC_DIR = add_sys_path(__file__, 1)
REPO_DIR = SRC_DIR.parent
PHYSBENCH_DATADIR = path(REPO_DIR, 'eval', 'physbench')


@dataclass
class One_PhysBench:
    file_names:List[str]
    question:str
    mode:str
    idx:int
    split:str


class PhysBenchData:
    def __init__(self, data_json=None):
        if not data_json:
            data_json = PHYSBENCH_DATADIR / 'test.json'
        self.data_list = auto_load(data_json)
        # print(len(self))
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        _cur_data = self.data_list[index]
        return One_PhysBench(
            file_names=_cur_data['file_name'],
            question=_cur_data['question'],
            mode=_cur_data['mode'],
            idx=_cur_data['idx'],
            split=_cur_data['split'],
        )

    def __iter__(self):
        def _func():
            for i in range(len(self)):
                yield self[i]
        return (_func())


class ModelToBeEvaluated:
    def __init__(self, model):
        self.model = model

    def qa(self, _one_piece:One_PhysBench) -> str:
        content = []
        file_needle = 0
        for _part in re.split(r'(<video>|<image>)', _one_piece.question):
            if _part == '<video>':
                _filename = PHYSBENCH_DATADIR / 'video' / _one_piece.file_names[file_needle]
                _filename = str(_filename)
                file_needle += 1
                content.append({
                    'type': 'video', 'video': _filename
                })
            elif _part == '<image>':
                _filename = PHYSBENCH_DATADIR / 'image' / _one_piece.file_names[file_needle]
                _filename = str(_filename)
                file_needle += 1
                content.append({
                    'type': 'image', 'image': _filename
                })
            else:
                content.append({
                    'type': 'text', 'text': _part
                })
        conversation = [{'role': 'user', 'content': content}]
        response = self.model(conversation)
        
        format_conversation = [{
            'role': 'user', 'content': f'''
# Reasoning Paragraph
{response}

# Instruction
Given the whole reasoning paragraph, conclude the output shortly.
Just output one single character: `A`, `B`, `C`, or `D`.
'''.strip()
        }]
        response = self.model(format_conversation)
        response = self.postprocess_reponse(response)
        return response
    
    @classmethod
    def postprocess_reponse(cls, response:str) -> str:
        def check(candidate:str):
            if len(candidate) == 1 and candidate.upper() in 'ABCD':
                return candidate.upper()
            elif len(candidate) == 2 and candidate[0].upper() in 'ABCD' and candidate[1] == '.':
                return candidate[0].upper()
            return None
        
        words = response.split()
        ans = check(words[0])
        if ans: return ans
        ans = check(words[-1])
        if ans: return ans
        print(f'> {response} < unformatted response')
        for i in range(len(words)-2, 0, -1):
            ans = check(words[i])
            if ans:
                return ans
        return None
    

def eval_physbench(model:ModelToBeEvaluated, model_name:str, just_val=True):
    all_res = []
    all_res_dic = {}
    for one_piece in PhysBenchData():
        if just_val and one_piece.split != 'val':
            continue
        response = model.qa(one_piece)
        all_res.append({'idx':one_piece.idx, 'answer':response})
        all_res_dic[one_piece.idx] = response
    
    auto_dump(all_res, SRC_DIR / 'results' / f'{model_name}.json')

    _ans_id_dic = dict(zip('ABCD', range(4)))
    pred, label = [], []
    for _val_label_dic in tqdm.tqdm(auto_load(PHYSBENCH_DATADIR/'val_answer.json')):
        if not _val_label_dic['answer']:
            continue
        pred.append(_ans_id_dic[all_res_dic[_val_label_dic['idx']]])
        label.append(_ans_id_dic[_val_label_dic['answer']])
    print(pred, label)
    import sklearn.metrics
    p,r,f,cnt = sklearn.metrics.precision_recall_fscore_support(label, pred)
    print(
        f'p: {p}\nr: {r}\nf: {f}\nlabels: {cnt}\nmacro-f1: {np.average(f):.4f}'
    )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    _model = lambda x: 'A'
    _model_name = 'all_A'
    _model = QwenVL(batch_output=False)
    _model_name = 'qwenvl_raw'
    _model = InternVL3_5()
    _model_name = 'internvl_raw'
    _model = LLaVA_NeXT_Video()
    _model_name = 'llavanv_raw'

    eval_physbench(
        model=ModelToBeEvaluated(_model),
        model_name=_model_name,
        just_val=True,
    )
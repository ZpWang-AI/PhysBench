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
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

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
        raw_response = response
        _output = self.postprocess_reponse(raw_response)
        if _output:
            return _output
        
        format_conversation = [{
            'role': 'user', 'content': [
                {'type': 'text', 'text': f'''
# Reasoning Paragraph
{response}

# Instruction
Given the whole reasoning paragraph, conclude the output shortly.
JUST output one single choice: `A.`, `B.`, `C.`, or `D.`.
DO NOT add any extra text or format decoration!
'''.strip()
                }
            ]
        }]
        response = self.model(format_conversation)
        formatted_response = response
        _output = self.postprocess_reponse(response)
        if not _output:
            FileIO.txt_dump(f'>> {self.model_name} <<\n{gap_line(fillchar="-")}\n{raw_response}\n{gap_line(fillchar="-")}\n{formatted_response}\n{gap_line()}\n', SRC_DIR/'~unformatted_response.txt', 'a')
        return _output
    
    # @classmethod
    def postprocess_reponse(self, response:str) -> str:
        def check_ABCD(candidate: str):
            match = re.match(r'^[A-D]$', candidate)
            if match:
                return match.group()

        def check2(candidate: str):
            match = re.match(r'^([A-D])\.$', candidate)
            if match:
                return match.group(1)

        def check3(candidate: str):
            match = re.match(r'^\W([A-D])(.)?\W$', candidate)
            if match:
                return match.group(1)

        response = response.strip()

        ans = check_ABCD(response)
        if ans: return ans

        words = response.split()

        ans = check2(words[0])
        if ans: return ans
        ans = check2(words[-1])
        if ans: return ans
        ans = check3(words[0])
        if ans: return ans
        ans = check3(words[-1])
        if ans: return ans

        if len(words) == 3 and re.match(r'^\W+$', words[0]) and re.match(r'^\W+$', words[2]):
            ans = check2(words[1])
            if ans: return ans
            ans = check3(words[1])
            if ans: return ans

        return None

        for i in range(len(words)-2, 0, -1):
            ans = check(words[i])
            if ans:
                return ans
        return None
    

def eval_physbench(model:ModelToBeEvaluated, model_name:str, just_val=True):
    save_filename = SRC_DIR / 'results' / f'{model_name}.json'
    all_res = auto_load(save_filename) if save_filename.exists() else []
    all_res_dic = {_dic['idx']:_dic['answer'] for _dic in all_res}
    for one_piece in tqdm.tqdm(PhysBenchData()):
        if just_val and one_piece.split != 'val':
            continue
        if one_piece.idx in all_res_dic and all_res_dic[one_piece.idx]:
            continue

        response = model.qa(one_piece)

        if one_piece.idx not in all_res_dic:
            all_res.append({'idx':one_piece.idx, 'answer':response})
            all_res.sort(key=lambda x:x['idx'])
        else:
            for _dic in all_res:
                if _dic['idx'] == one_piece.idx:
                    _dic['answer'] = response
        all_res_dic[one_piece.idx] = response
        auto_dump(all_res, save_filename)


    _ans_id_dic = dict(zip('ABCD', range(4)))
    _ans_id_dic[None] = 4
    pred, label = [], []
    for _val_label_dic in auto_load(PHYSBENCH_DATADIR/'val_answer.json'):
        if not _val_label_dic['answer']:
            continue
        _idx, _ans = _val_label_dic['idx'], _val_label_dic['answer']
        pred.append(_ans_id_dic[all_res_dic[_idx]])
        label.append(_ans_id_dic[_ans])
    # print(pred, label)
    import sklearn.metrics
    acc = sklearn.metrics.accuracy_score(label, pred)
    print(acc, len(pred)*acc, len(pred))
    return
    p,r,f,cnt = sklearn.metrics.precision_recall_fscore_support(label, pred)
    print(
        f'p: {p}\nr: {r}\nf: {f}\nlabels: {cnt}\nmacro-f1: {np.average(f):.4f}'
    )


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
        model=ModelToBeEvaluated(_model, _model_name),
        model_name=_model_name,
        just_val=True,
    )
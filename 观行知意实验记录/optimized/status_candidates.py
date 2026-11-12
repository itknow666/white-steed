"""只读查看本轮训练队列、当前进度与已完成模型。"""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'runs/candidates-v6-20261106'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
status=read(out/'progress.json')
result={'queue':{k:v for k,v in status.items() if k!='command'}}
if status.get('model'):
    run=root/'runs'/f'{status["model"]}-v6-20261106'
    if (run/'progress.json').exists():result['training']=read(run/'progress.json')
    if (run/'history.jsonl').exists():result['last_validation']=json.loads((run/'history.jsonl').read_text().splitlines()[-1])
if (out/'results-partial.json').exists():result['completed']=read(out/'results-partial.json')
if (out/'outcome.json').exists():result['outcome']=read(out/'outcome.json')
print(json.dumps(result,ensure_ascii=True,indent=2))

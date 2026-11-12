"""Summarize validation predictions; add the public manifest for class metrics."""
import argparse,csv,hashlib,html,json
from collections import Counter
from pathlib import Path

def analyze(predictions,manifest=None):
    with predictions.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('预测文件为空')
    values={}
    for row in rows:
        sample=row.get('sample_id','');raw=row.get('predicted_class','')
        if not sample or sample in values:raise ValueError('sample_id 为空或重复')
        try:value=int(raw)
        except (TypeError,ValueError):raise ValueError('predicted_class 必须为整数') from None
        if not 0<=value<40:raise ValueError('predicted_class 必须在0—39之间')
        values[sample]=value
    counts=Counter(values.values())
    result={'samples':len(values),'predicted_classes':len(counts),
            'prediction_counts':{str(i):counts[i] for i in range(40)},
            'largest_prediction_share':max(counts.values())/len(values),
            'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
            'label_source':None,'metrics':None}
    if manifest is None:return result
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('modality')!='depth_ir' or len(data.get('classes',[]))!=40:
        raise ValueError('manifest 必须为40类公开深度数据')
    truth={}
    for sample in data['samples']:
        name,label=sample.get('sample_id',sample['file']+'.npy'),sample['label']
        if name in truth or type(label) is not int or not 0<=label<40:raise ValueError('manifest 样本重复或标签无效')
        truth[name]=label
    if truth.keys()!=values.keys():
        raise ValueError(f'样本不匹配：缺失{len(truth.keys()-values.keys())}，多余{len(values.keys()-truth.keys())}')
    matrix=[[0]*40 for _ in range(40)];errors=[]
    for name,label in truth.items():
        prediction=values[name];matrix[label][prediction]+=1
        if label!=prediction:errors.append({'sample_id':name,'label':label,'predicted_class':prediction,
                                           'true_name':data['classes'][label],'predicted_name':data['classes'][prediction]})
    per_class=[]
    for i in range(40):
        tp=matrix[i][i];support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
        per_class.append({'class_id':i,'class_name':data['classes'][i],'support':support,'predicted':predicted,
                          'precision':tp/predicted if predicted else 0,
                          'recall':tp/support if support else 0,
                          'f1':2*tp/(support+predicted) if support+predicted else 0})
    result.update(label_source=str(manifest),classes=data['classes'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  metrics={'accuracy':100*sum(matrix[i][i] for i in range(40))/len(values),
                           'macro_f1':100*sum(row['f1'] for row in per_class)/40},
                  confusion_matrix=matrix,per_class=per_class,errors=errors)
    return result

def render(data):
    metric=data['metrics']
    lead=f"{data['samples']} 条预测，输出涉及 {data['predicted_classes']} 个类别。"
    if data['predicted_classes']==1:lead+='全部预测为同一类别，应结合训练曲线、标签分布和输入预处理检查。'
    metrics=(f"准确率 {metric['accuracy']:.2f}% · Macro F1 {metric['macro_f1']:.2f}%（40类平均）" if metric else '当前仅分析预测分布。加入公开验证集 manifest.json 后生成准确率、Macro F1 和错例。')
    names=data.get('classes',[str(i) for i in range(40)])
    rows=''.join(f'<tr><td>{html.escape(names[i])}</td><td>{data["prediction_counts"][str(i)]}</td></tr>' for i in range(40))
    extra=''
    if metric:
        extra='<h2>类别表现</h2><table><tr><th>类别</th><th>样本数</th><th>召回率</th><th>F1</th></tr>'+''.join(f'<tr><td>{html.escape(x["class_name"])}</td><td>{x["support"]}</td><td>{x["recall"]:.3f}</td><td>{x["f1"]:.3f}</td></tr>' for x in data['per_class'])+'</table>'
        extra+='<h2>错例</h2><table><tr><th>样本</th><th>真实动作</th><th>预测动作</th></tr>'+''.join(f'<tr><td>{html.escape(x["sample_id"])}</td><td>{html.escape(x["true_name"])}</td><td>{html.escape(x["predicted_name"])}</td></tr>' for x in data['errors'])+'</table>'
    return '<!doctype html><meta charset="utf-8"><title>验证集预测分析</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#203044}table{border-collapse:collapse;width:100%}td,th{padding:6px 12px;text-align:left;border-bottom:1px solid #ddd}h2{margin-top:32px}</style><h1>验证集预测分析</h1><p>'+lead+'</p><p>'+metrics+'</p><h2>预测分布</h2><table><tr><th>类别编号</th><th>预测数量</th></tr>'+rows+'</table>'+extra

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path,default=Path('validation-report'))
    args=parser.parse_args();result=analyze(args.predictions,args.manifest)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.html').write_text(render(result),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k in ('samples','predicted_classes','largest_prediction_share','metrics')},ensure_ascii=False))

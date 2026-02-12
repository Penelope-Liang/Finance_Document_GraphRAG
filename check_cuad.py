import json

with open('cuad/data/train_separate_questions.json', 'r') as f:
    data = json.load(f)

print('=== CUAD 本地数据统计 ===')
print(f'合同总数: {len(data["data"])}')

lengths = []
total_qas = 0
for i, article in enumerate(data['data']):
    title = article['title']
    context = article['paragraphs'][0]['context']
    qas_count = len(article['paragraphs'][0]['qas'])
    lengths.append((i+1, title[:40], len(context), qas_count))
    total_qas += qas_count

print(f'问题总数: {total_qas}')
print(f'平均每份合同问题数: {total_qas / len(data["data"]):.1f}')

# 按长度排序
lengths.sort(key=lambda x: x[2], reverse=True)

print(f'\nTop 5 最长合同:')
for idx, title, length, qas in lengths[:5]:
    print(f'  #{idx} {title}...: {length:,} 字符, {qas} 问题')

print(f'\nTop 5 最短合同:')
for idx, title, length, qas in lengths[-5:]:
    print(f'  #{idx} {title}...: {length:,} 字符, {qas} 问题')

all_lens = [x[2] for x in lengths]
print(f'\n长度统计:')
print(f'  最短: {min(all_lens):,} 字符')
print(f'  最长: {max(all_lens):,} 字符')
print(f'  平均: {sum(all_lens)//len(all_lens):,} 字符')

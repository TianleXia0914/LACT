import pandas
import re
import os
import pickle    
import jsonlines
from tqdm import tqdm
tsv_path=r'../data/mid2wikipedia.tsv'
df=pandas.read_csv(tsv_path, sep='\t')
mid2en_name={}
for mid,en_name in zip(df['mid'],df['en_name']):
    mid2en_name[mid]=en_name
def get_relation(number,id2relation):
    digit=int(number)
    str_relation=id2relation[digit]
    if '+' in str_relation:
        parts = str_relation.split('/')
        str_relation=f'{parts[-2]} {parts[-1]}'
    elif '-' in str_relation:
        parts = str_relation.split('/')
        str_relation=f'the reverse relationship of {parts[-2]} {parts[-1]}'
    else:
        str_relation=str_relation[8:]
        str_relation=str_relation.replace('_',' ')
    return str_relation
        
def get_entity(number,id2entity):
    digit=int(number)
    try:
        en_name=id2entity[digit]
        if not 'concept' in en_name:
            if("," in en_name):
                en_name=number
            else:
                try:
                    en_name=mid2en_name[en_name]
                except:
                    en_name=number
        else:
            en_name=en_name[8:]
            en_name=en_name.replace("_"," ")
    
            
    except:
        en_name=number
    #print(en_name)
    return en_name        
'''  
def get_entity(number,id2entity):
    digit=int(number)
    mid=id2entity[digit]
    if('concept' in mid):
        en_name=en_name[8:]
    try:
        en_name=mid2en_name[mid]
    except:
        en_name=number
    return en_name
'''
def trans_relation_num(text,id2relation):   
    matches = re.findall(r'relation \d+', text)
    for relation in matches:  
        number=relation[9:]
        data=get_relation(number,id2relation)
        data='relation '+data
        text = text.replace(relation, data)
    #triplets = re.findall('\((\d+),(\d+),(\d+)\)', text)
    matches = re.findall(r'other than \d+', text)
    for relation in matches:  
        number=relation[11:]
        data=get_relation(number,id2relation)
        data='other than '+data
        text = text.replace(relation, data)
    triplets = re.findall(',\d+,', text)
    for triplet in triplets:
        numbers = re.findall(r'\d+', triplet)
        #relation_num=numbers[1]
        relation_num=numbers[0]
        data=get_relation(relation_num,id2relation)
        '''
        if(data[0]=='+'):
            a=f'({numbers[0]},{data[1:]},{numbers[2]})'
        else:
            a=f'({numbers[2]},{data[1:]},{numbers[1]})'
        '''
        a=f",{data},"
        text=text.replace(triplet,a)
    return text
def trans_entity_num(text,id2entity):
    matches = re.findall(r'entity \d+', text)
    for entity in matches:  
        number=entity[7:]
        data=get_entity(number,id2entity)
        a='entity '+data
    
        text = text.replace(entity, a)
    matches = re.findall(r'connected to \d+', text)
    for entity in matches:  
        number=entity[13:]
        data=get_entity(number,id2entity)
        a='connected to '+data
    
        text = text.replace(entity, a)
    #triplets = re.findall('\((\d+),(\d+),(\d+)\)', text)
    triplets = re.findall('\(\d+,\d+,\d+\)', text)
    for triplet in triplets:
        numbers = re.findall(r'\d+', triplet)
        h=numbers[0]
        s=numbers[2]
        datah=get_entity(h,id2entity)
        datat=get_entity(s,id2entity)
        a=f'({datah},{numbers[1]},{datat})'
        
        text=text.replace(triplet,a)
    return text
def remove_strip(text):
    list=text.split(' ')
    listnew=[i for i in list if i!='' ]
    new=' '.join(listnew)
    return new
def trans_answer(text,id2entity,id2relation):
    index=text.find('the final answer is')
    pre_text=text[:index]
    answer_text=text[index:]
    numbers=re.findall(r'\d+',answer_text)
    temp=[]
    for number in numbers:
        data=get_entity(number,id2entity)
        temp.append(data)
    s=','.join(temp)
    answer_text='the final answer is '+s
    matches = re.findall(r'entity \d+', pre_text)
    for entity in matches:  
        number=entity[7:]
        data=get_entity(number,id2entity)
        a='entity '+data
        pre_text = pre_text.replace(entity, a)
    matches = re.findall(r'connected to \d+', pre_text)
    for entity in matches:  
        number=entity[13:]
        data=get_entity(number,id2entity)
        a='connected to '+data
        pre_text = pre_text.replace(entity, a)
    matches = re.findall(r'relation \d+', pre_text)
    for relation in matches:  
        number=relation[9:]
        data=get_relation(number,id2relation)
        data='relation '+data
        pre_text = pre_text.replace(relation, data)
    matches = re.findall(r'other than \d+', pre_text)
    for relation in matches:  
        number=relation[11:]
        data=get_relation(number,id2relation)
        data='other than '+data
        pre_text = pre_text.replace(relation, data)
    s=pre_text+answer_text
    return s

def main(data_path, output_path):
    t=0
    data_root=data_path
    new_data_root=output_path
    for dataset_name in os.listdir(data_root):
        dataset_path=os.path.join(data_root,dataset_name)
        output_dataset_path=os.path.join(new_data_root,dataset_name)
        origin_data_root=f'../data/{dataset_name}-betae'
        id2rel_path=os.path.join(origin_data_root,'id2rel.pkl')
        id2ent_path=os.path.join(origin_data_root,'id2ent.pkl')
        with open(id2rel_path,'rb') as f:
            id2relation=pickle.load(f)
        with open(id2ent_path,'rb')as f:
            id2entity=pickle.load(f)
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)
        for data_name in os.listdir(dataset_path):
            data_path=os.path.join(dataset_path,data_name)
            output_data_path=os.path.join(output_dataset_path,data_name)
            print(output_data_path)
            with jsonlines.open(output_data_path,'w')as nf:
                with open(data_path,'r')as f:
                    for item in tqdm(jsonlines.Reader(f)):
                        instruction=item['instruction']
                        output=item['output']
                        instruction=trans_entity_num(instruction,id2entity)
                        output=trans_answer(output,id2entity,id2relation)
                        instruction=trans_relation_num(instruction,id2relation)
                        
                        instruction=remove_strip(instruction)
                        output=remove_strip(output)
                        temp={
                            "instruction":instruction,
                            "input":"",
                            "output":output,
                            "demonstrations":"",
                        }
                        nf.write(temp)
        
         
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to raw data.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output the processed files.")
    args = parser.parse_args()
    main(args.data_path, args.output_path)
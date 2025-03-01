import os
import jsonlines
import re
def main(data_path, output_path):
    nums={"1p":1,"2i":3,"2in":3,"2p":2,"3i":4,"3in":4,"3p":3,"inp":4,"pin":4,"pni":4}
    output_data_root=data_path
    origin_data_root=output_path
    for dateset_name in os.listdir(origin_data_root):
        origin_dataset_path=os.path.join(origin_data_root,dateset_name)
        output_dataset_path=os.path.join(output_data_root,dateset_name)
        if not os.path.exists(output_dataset_path):
            os.makedirs(output_dataset_path)
        for data_name in os.listdir(origin_dataset_path):
            origin_data_path=os.path.join(origin_data_root,dateset_name,data_name)
            output_data_path=os.path.join(output_data_root,dateset_name,data_name)
            with jsonlines.open(output_data_path,'w')as nf:
                with open(origin_data_path,'r')as f:
                    print(data_name)
                    
                    for item in jsonlines.Reader(f):
                        answer=item['answer']
                        task_type=data_name[:-6]
                        #step_query=item['step']
                        instruction=item['question']
                        index1=instruction.find('\nReturn only the answer entities separated by commas with no other text')
                        instruction=instruction[:index1]
                        index=instruction.find('question:')
                        #premise=step_query[8:index] #8 is the length of premise
                        questions=instruction[index+9:]
                        #questions=eval(questions)
                        #question=questions['question']
                        #k=len(question[task_type])
                        numbers = re.findall(r'\d+', questions)
                        k=nums[task_type]
                        output=f"Let's think step by step.The question can be split into {k} question.\n"
                        if(task_type=='1p'):
                            question=[questions]
                            for i,data in enumerate(question):
                                output+=f"{i+1}.{data}\n"
                                output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='2i'):
                            entity1,relation1,entity2,relation2=numbers
                            question={"2i": [f"Which entities are connected to {entity1} by relation {relation1}?",
                            f"Which entities are connected to {entity2} by relation {relation2}?",
                            f"What are the entities in the intersection of entity sets [PP1] and [PP2]?"]
                            }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='2in'):
                            entity1,relation1,entity2,relation2=numbers
                            question={"2in": [f"Which entities are connected to {entity1} by relation {relation1}?",
                            f"Which entities are connected to {entity2} by any relation other than {relation2}?",
                            f"What are the entities in the intersection of entity sets [PP1] and [PP2]?"]
                            }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='2p'):
                            entity,relation1,relation2=numbers
                            question={"2p": [f"Which entities are connected to {entity} by relation {relation1}?",
                        f"Which entities are connected to any entity in [PP1] by relation {relation2}?"]
                    }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='2u'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='3i'):
                            entity1,relation1,entity2,relation2,entity3,relation3=numbers
                            question={"3i": [f"Which entities are connected to {entity1} by relation {relation1}?",
                        f"Which entities are connected to {entity2} by relation {relation2}?",
                        f"Which entities are connected to {entity3} by relation {relation3}?",
                        f"What are the entities in the intersection of entity sets [PP1], [PP2] and [PP3]?"]
                            }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='3in'):
                            entity1,relation1,entity2,relation2,entity3,relation3=numbers
                            question={"3in": [f"Which entities are connected to {entity1} by any relation other than {relation1}?",
                            f"Which entities are connected to {entity2} by relation  {relation2}?",
                            f"Which entities are connected to {entity3} by relation {relation3}?",
                            f"What are the entities in the intersection of entity sets [PP1], [PP2] and [PP3]?"]
                    }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='3p'):
                            entity,relation1,relation2,relation3=numbers
                            question={"3p": [f"Which entities are connected to {entity} by relation {relation1}?",
                        f"Which entities are connected to any entity in [PP1] by relation {relation2}?",
                        f"Which entities are connected to any entity in [PP2] by relation {relation3}?"]
                    }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='inp'):
                            entity1,relation1,entity2,relation2,relation3=numbers
                            question={"inp": [f"Which entities are connected to {entity1} by relation {relation1}?",
                            f"Which entities are connected to {entity2} by any relation other than {relation2}?",
                            f"What are the entities in the intersection of entity sets [PP1], and [PP2]?",
                            f"What are the entities connected to any entity in [PP3] by relation {relation3}?"]
                    } 
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='ip'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='nin'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='nipn'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='pi'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='pin'):
                            entity1,relation1,relation2,entity2,relation3=numbers
                            question={"pin": [f"Which entities are connected to {entity1} by relation {relation1}?",
                            f"Which entities are connected to entity set in [PP1] by relation {relation2}?",
                            f"Which entities are connected to {entity2} by any relation other than {relation3}?",
                            f"What are the entities in the intersection of entity sets [PP2] and [PP3]?"]
                            }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='pni'):
                            entity1,relation1,relation2,entity2,relation3=numbers
                            question={"pni": [f"Which entities are connected to {entity1} by relation {relation1}?",
                            f"Which entities are connected to any entity in [PP1] by any relation other than {relation2}?",
                            f"Which entities are connected to {entity2} by relation {relation3}?",
                            f"What are the entities in the intersection of entity sets [PP2] and [PP3]?"]
                            }
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        elif(task_type=='up'):
                            for i,data in enumerate(question[task_type]):
                                output+=f"{i+1}.{data}\n"
                                if(i==0):
                                    output+='The entity set of the answer is represented by [PP1].\n'
                                elif(i==1):
                                    output+='The entity set of the answer is represented by [PP2].\n'
                                elif(i==2):
                                    output+='The entity set of the answer is represented by [PP3].\n'
                                else:
                                    output+='With reference to the relevant triplet above,the final answer is '
                        output+=answer
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
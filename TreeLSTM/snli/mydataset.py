from visual_genome import api
import json
from torchtext import data
import os

# ids = api.get_all_image_ids()
# print(ids[0])
#graph = api.get_scene_graph_of_image()
class node(object):
    def __init__(self, value, children = []):
        self.value = value
        self.children = children
    def __repr__(self, level=0):
        return self.value

class mydataset():
    def __init__(self,path,device,data_format='json'):
        self.dataset_path = path
        self.data_format = data_format
        self.device = device
        self.dict_dataset = []
        self.sentences = []
    def write_dataset(self):
        data_format = self.data_format.lower()
        # dict_dataset = [
        #     {"id": "0", "question1": "When do you use シ instead of し?",
        #      "question2": "When do you use \"&\" instead of \"and\"?",
        #      "label": "0"},
        #     {"id": "1", "question1": "Where was Lincoln born?",
        #      "question2": "Which location was Abraham Lincoln born?",
        #      "label": "1"},
        #     {"id": "2", "question1": "What is 2+2",
        #      "question2": "2+2=?",
        #      "label": "1"},
        # ]
        #sentence = {"premise": "( ( The church ) ( ( has ( cracks ( in ( the ceiling ) ) ) ) . ) )"}
        #ids = api.get_all_image_ids()

        regions = api.get_region_descriptions_of_image(id=61512)#id=61512
        for item in regions:
            sentence = {"premise": item.phrase}
            self.dict_dataset.append(sentence)
            self.sentences.append(item.phrase.split(' '))
        regions = api.get_region_descriptions_of_image(id=1)  # id=61512
        for item in regions:
            sentence = {"premise": item.phrase}
            self.dict_dataset.append(sentence)
            self.sentences.append(item.phrase.split(' '))
        sentence = "The church has cracks in the ceiling ."
        sentenceDict = {"premise": "The church has cracks in the ceiling ."}
        self.dict_dataset.append(sentenceDict)
        self.sentences.append(sentence.split(' '))
        if not os.path.exists(self.test_ppid_dataset_path):
            with open(self.dataset_path, "w") as file:
                for example in self.dict_dataset:
                    if data_format == "json":
                        file.write(json.dumps(example) + "\n")
                    else:
                        raise ValueError("Invalid format {}".format(data_format))

    def build_dateset(self):
        question_field = data.Field(sequential=True)
        # fields = {"question1": ("q1", question_field),
        #           "question2": ("q2", question_field),
        #           "label": ("label", label_field)}
        fields = {"premise": ("premise", question_field)}
        test = data.TabularDataset(
            path=self.dataset_path,format=self.data_format, fields=fields)
        # print(test[0].__dict__.keys())
        # print(test[0].premise)
        #print(dataset[1].premise)
        question_field.build_vocab(test)
        iter = data.BucketIterator(test, batch_size=128, device=self.device)
        return iter
    def get_sentences(self):
        return self.sentences
if __name__ == "__main__":

    mydataset = mydataset('test.json','0')
    mydataset.write_dataset()
    aa = mydataset.get_sentences()
    print(aa[0])
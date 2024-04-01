import json

class JSONLReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        data_list = []
        with open(self.file_path, 'r',encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)
        return data_list

    # def process_data(self, data):
    #     source = data['source']  # Source of the citation
    #     cite_start = data['citeStart']  # Start position of the citation
    #     cite_end = data['citeEnd']  # End position of the citation
    #     section_name = data['sectionName']  # Name of the section containing the citation
    #     citation_string = data['string']  # Actual citation text
    #     label = data['label']  # Label assigned to the citation
    #     label_confidence = data['label_confidence']  # Confidence level of the assigned label
    #     citing_paper_id = data['citingPaperId']  # ID of the citing paper
    #     cited_paper_id = data['citedPaperId']  # ID of the cited paper
    #     is_key_citation = data['isKeyCitation']  # Whether the citation is a key citation
    #     unique_id = data['unique_id']  # Unique identifier for the citation
    #     excerpt_index = data['excerpt_index']  # Index of the citation excerpt
    #
    #     # Extract the text from the citation string
    #     text = citation_string.strip()  # Remove leading and trailing whitespace
    #
    #     return {
    #         'source': source,
    #         'cite_start': cite_start,
    #         'cite_end': cite_end,
    #         'section_name': section_name,
    #         'citation_string': citation_string,
    #         'label': label,
    #         'label_confidence': label_confidence,
    #         'citing_paper_id': citing_paper_id,
    #         'cited_paper_id': cited_paper_id,
    #         'is_key_citation': is_key_citation,
    #         'unique_id': unique_id,
    #         'excerpt_index': excerpt_index,
    #         'text': text
    #     }
    #
    # def process_file(self):
    #     data_list = self.read_file()
    #     processed_data_list = []
    #     for data in data_list:
    #         processed_data = self.process_data(data)
    #         processed_data_list.append(processed_data)
    #     return processed_data_list


if __name__ == '__main__':
    # Example usage
    file_path = '../../data/train.jsonl'
    reader = JSONLReader(file_path)
    processed_data_list = reader.process_file()

    for data in processed_data_list:
        print(data)
    file_path = '../../data/train.jsonl'
    reader = JSONLReader(file_path)
    processed_data_list = reader.process_file()


    for data in processed_data_list:
        print(data)

import json
from pathlib import Path


qa_balanced = 'balanced_qa_pairs.json'
qa_generated = '_qa_pairs.json'


def main():
    qa_b = json.load(open(Path(__file__).parent / qa_balanced))
    qa_g = json.load(open(Path(__file__).parent / qa_generated))
    print(f"Number of qa_pairs golden: {len(qa_b)}")
    print(f"Number of qa_pairs generated: {len(qa_g)}")

    # if every qa_pair in qa_b also exists in qa_g otherwise print it out
    count = 0
    correct = 0
    for idx, qa in enumerate(qa_b):
        found = False
        for qb in qa_g:
            # print("Comparing: " + str(qb["question"]) + " to " + str(qa["question"]))
            if (qb["question"] == qa["question"] and qb["image_file"] == qa["image_file"]):
                found = True
                if qb["answer"] == qa["answer"]:
                    correct += 1
                else:
                    print(
                        f"Wrong answer: {qa}   \nCorrect answer: {qb['answer']}")
                break
        if not found:
            print("Not found: ", qa)
            count += 1
    print(f"Number of qa_pairs missing: {count}  Correct: {correct}")


main()

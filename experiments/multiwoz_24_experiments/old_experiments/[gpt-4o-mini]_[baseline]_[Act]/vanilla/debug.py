import pickle
import collections

pkl_path = "experiments/multiwoz_24_experiments/[gpt-4o-mini]_[baseline]_[Act]/vanilla/total_record.pkl"

with open(pkl_path, "rb") as f:
    total_record = pickle.load(f)

system_responses = [turn.system_response for dialogue in total_record["test"].values() for turn in dialogue.values()]
lengths = [len(response) for response in system_responses]
print(f"Number of responses: {len(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Average length: {sum(lengths) / len(lengths):.2f}")
print("Length distribution (top 10):", collections.Counter(lengths).most_common(10))
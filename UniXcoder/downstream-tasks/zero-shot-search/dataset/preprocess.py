import json

for lang,suffix in [("Java",".java"),("Ruby",".rb"),("Python",".py")]:
    with open("{}.jsonl".format(lang.lower())) as f, open("{}_with_func.jsonl".format(lang.lower()),"w") as f1:
        for line in f:
            js = json.loads(line.strip())
            problem_id = str(js["label"])
            problem_id = "p" + "0" * (5-len(problem_id)) + problem_id
            language = lang
            submission_id = js["index"]
            func = open("Project_CodeNet/data/{}/{}/{}{}".format(problem_id,language,submission_id,suffix)).read()
            js["func"] = func
            f1.write(json.dumps(js)+"\n")

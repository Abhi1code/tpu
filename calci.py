import json

def generate_all_cat(in_data):
    # Opening JSON file
    f = open('data_cats.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    # print(data["categories"])
    # print(len(data["categories"]))

    # Closing file
    f.close()

    all_categories = {}
    rev_all_cat = {}

    for i in range(len(data["categories"])):
        all_categories[data["categories"][i]["name"]] = data["categories"][i]["id"]
        rev_all_cat[data["categories"][i]["id"]] = data["categories"][i]["name"]

    all_attributes = {}
    rev_all_attr = {}

    for i in range(len(data["attributes"])):
        if data["attributes"][i]["supercategory"] not in all_attributes.keys():
            all_attributes[data["attributes"][i]["supercategory"]] = {}

        all_attributes[data["attributes"][i]["supercategory"]][data["attributes"][i]["id"]] = data["attributes"][i][
            "name"]
        rev_all_attr[data["attributes"][i]["id"]] = data["attributes"][i]["name"]

    data_res = []
    data_res.append(in_data)

    cat_count = {}

    for i in range(46):
        cat_count[i] = 0

    for i in range(len(data_res)):
        for j in range(len(data_res[i]["classes"])):
            if data_res[i]["scores"][j] > 0.2:
                cat_count[data_res[i]["classes"][j]] += 1

    temp = sorted(cat_count.items(), key=lambda kv: (kv[1], kv[0]))
    temp = temp[-20:]

    choosed_categories = []

    for i in range(len(temp)):
        choosed_categories.append(temp[i][0])

    choosed_attributes = {}  # all for now

    for super_attr in (all_attributes):
        for attr in all_attributes[super_attr]:
            choosed_attributes[attr] = super_attr

    prevs_structure = {}
    our_structure = prevs_structure

    for i in range(len(choosed_categories)):
        our_structure[choosed_categories[i]] = {}
        for j in range(len(choosed_attributes)):
            # print(choosed_attributes[j]," ",our_structure[choosed_categories[i]] )
            if choosed_attributes[j] not in our_structure[choosed_categories[i]]:
                our_structure[choosed_categories[i]][choosed_attributes[j]] = {}
            our_structure[choosed_categories[i]][choosed_attributes[j]][j] = 0
            # our_structure[choosed_categories[i]][choosed_attributes[j]] = 0

    Cat_threshold = 0.20
    atrr_threshold = 0.20

    ttt = {}

    for no in range(len(data_res)):

        cat_result = []
        for i in range(len(data_res[no]["classes"])):
            if data_res[no]["scores"][i] > Cat_threshold and data_res[no]["classes"][i] in choosed_categories:
                cat_result.append(data_res[no]["classes"][i])

        for i in range(len(data_res[no]["classes"])):
            if data_res[no]['classes'][i] in cat_result:
                temp = {}
                for j in range(len(data_res[no]["attributes"][i])):
                    if data_res[no]["attributes"][i][j] > atrr_threshold and j in choosed_attributes:
                        our_structure[data_res[no]["classes"][i]][choosed_attributes[j]][j] += 1
                        temp[rev_all_attr[j]] = 1

                ttt[rev_all_cat[data_res[no]["classes"][i]]] = temp
    return ttt










def test(dataset_questions_path, features_path, BATCH_SIZE,
         lstm, rn, criterion, questions_dictionary, answers_dictionary, device,
         MAX_QUESTION_LENGTH, isObjectFeatures, OBJECT_TRIM="default"):

    val_loss = 0.
    val_accuracy = 0.

    set_eval_mode(rn)
    set_eval_mode(lstm)

    with torch.no_grad():

        dataset_size_remain = get_size(dataset_questions_path)

        print("Testing")
        batch = get_batch(dataset_questions_path, features_path, BATCH_SIZE,
                          device, isObjectFeatures, categoryBatch=True, OBJECT_TRIM=OBJECT_TRIM)

        groups = {}
        groups_acc = []
        types = {"semantic":   {},
                 "detailed":   {},
                 "structural": {}
                 }
        semantic_acc = []
        structural_acc = []
        detailed_acc = []

        #pbar = tqdm(total=num_batch)
        batch_number = 0
        while dataset_size_remain > 0:

            # Get batch

            if dataset_size_remain < BATCH_SIZE:
                break
            dataset_size_remain -= BATCH_SIZE

            #question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = next(batch)
            question_batch, answer_ground_truth_batch, object_features_batch, category_batch = next(
                batch)

            h_q = lstm.reset_hidden_state()

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                                      questions_dictionary, answers_dictionary,
                                                                      BATCH_SIZE, device, MAX_QUESTION_LENGTH)

            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(
                question_batch, h_q)
            question_emb_batch = question_emb_batch[:, -1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch)

            loss = criterion(rr, answer_ground_truth_batch)
            val_loss += loss.item()

            correct, _, correct_answers = get_answer(
                rr, answer_ground_truth_batch, answers_dictionary, return_answer=True)
            val_accuracy += correct

            """
            Structure:
                groups -> {
                    "group_name1": (100, 200), -> This means 200 question were tested and 100 were correct, giving 50% accuracy
                    "group_name2": (20, 30),
                    "group_name3": (5, 8)
                }
                
                types -> {
                    "structural": {
                        "struct1": (20, 30), -> same structure as the one on groups
                        "struct2": (10, 15),
                    },
                    "semantic": {}, -> they might be empty (so might "groups")
                    "detailed": {
                        "det1": (1, 4),
                        "det2": (30, 45)
                    }
                }
            """

            # Obtain results for each group and type
            for question, correct_answer in zip(category_batch, correct_answers):

                group = question["group"]  # e.g. -> all color questions
                if group is not None:
                    group_rights, group_total = groups.get(
                        group, (0, 0))  # groups[group] = (0,0)
                    groups[group] = (
                        group_rights + correct_answer, group_total + 1)
                else:
                    group_rights, group_total = groups.get("None", (0, 0))
                    groups["None"] = (
                        group_rights + correct_answer, group_total + 1)

                # -> e.g. semantic, detailed, structural
                for typ in question["types"]:
                    type_category = question["types"][typ]  # -> e.g. query
                    if type_category is not None:
                        category_rights, category_total = types[typ].get(
                            type_category, (0, 0))
                        types[typ][type_category] = (
                            category_rights + correct_answer, category_total + 1)
                    else:
                        category_rights, category_total = types[typ].get(
                            "None", (0, 0))
                        types[typ]["None"] = (
                            category_rights + correct_answer, category_total + 1)

            batch_number += 1
            #pbar.update()

        print(f"Accuracy seperated by group")
        for group in groups:
            rights, total = groups[group]
            groups_acc.append([group, 100*rights/total])
            print(
                f"Group: {group} -> {rights}/{total} gives us {100*rights/total}% ")
        write_csv(groups_acc, "group_accuracy")

        print("___________________________________")

        print(f"Accuracy seperated by types")
        for typ in types:
            print(f"Type: {typ}")
            current_type = types[typ]
            for category in current_type:
                rights, total = current_type[category]
                print(
                    f"Category: {category} -> {rights}/{total} gives us {100*rights/total}% ")
            print("___________________________________")

        for category in types["semantic"]:
            rights, total = types["semantic"][category]
            semantic_acc.append([category, 100*rights/total])
        write_csv(semantic_acc, "semantic_accuracy")

        for category in types["structural"]:
            rights, total = types["structural"][category]
            structural_acc.append([category, 100*rights/total])
        write_csv(structural_acc, "structural_accuracy")

        for category in types["detailed"]:
            rights, total = types["detailed"][category]
            detailed_acc.append([category, 100*rights/total])
        write_csv(detailed_acc, "detailed_accuracy")

        #pbar.close()

        val_accuracy /= float(batch_number)
        val_loss /= float(batch_number)

        return val_loss, val_accuracy

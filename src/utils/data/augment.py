class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, tags, attention_mask, mention_positions, auxiliary_datasets, transform=None):
        self.input_ids = input_ids
        self.tags = tags
        self.attention_mask = attention_mask
        self.mention_positions = mention_positions
        self.auxiliary_datasets = auxiliary_datasets
        self.transform = transform

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        sample = self.input_ids[idx], self.tags[idx], self.attention_mask[idx]

        if self.transform:
            sample = self.transform(sample, self.mention_positions[idx], self.auxiliary_datasets)

        return sample


class MentionSwapAugment(object):

    def __init__(self, chance, o_id, i_id):
        self.chance = chance
        self.o_id = o_id
        self.i_id = i_id

    def __call__(self, sample, mention_positions, auxiliary_datasets):
        input_ids, tags, attention_mask = sample

        if random.random() <= self.chance and len(mention_positions) > 0:
            for ment_pos in mention_positions:
                if len(ment_pos) > 1:
                    new_dataset = auxiliary_datasets[random.randrange(0, len(auxiliary_datasets))]

                    new_dataset_len = len(new_dataset)
                    if len(new_dataset) == len(ment_pos):
                        input_ids[ment_pos] = new_dataset
                    else:
                        length_diff = len(ment_pos) - new_dataset_len
                        #                         print(tokenizer.convert_ids_to_tokens(input_ids))
                        #                         print(tags)
                        #                         print(attention_mask)
                        #                         print(ment_pos)

                        if length_diff > 0:
                            return self.augment_with_smaller_label(input_ids, tags, attention_mask, new_dataset,
                                                                   ment_pos, length_diff, new_dataset_len)
                        else:
                            if torch.count_nonzero(input_ids) - length_diff <= len(input_ids):
                                return self.augment_with_bigger_label(input_ids, tags, attention_mask, new_dataset,
                                                                      ment_pos, length_diff, new_dataset_len)

        return input_ids, tags, attention_mask

    def augment_with_smaller_label(self, input_ids, tags, attention_mask, new_dataset, mention_positions, length_diff,
                                   new_dataset_len):
        # Set new tokens
        new_token_positions = mention_positions[:new_dataset_len]
        input_ids[new_token_positions] = new_dataset

        # Roll back rest of sentence and tags
        input_ids[new_token_positions[-1] + 1:-length_diff] = input_ids[
                                                              new_token_positions[-1] + 1 + length_diff:].clone()
        tags[new_token_positions[-1] + 1:-length_diff] = tags[new_token_positions[-1] + 1 + length_diff:].clone()
        attention_mask[new_token_positions[-1] + 1:-length_diff] = attention_mask[
                                                                   new_token_positions[-1] + 1 + length_diff:].clone()

        # Fill ends (padding and O token)
        input_ids[-length_diff:] = 0
        tags[-length_diff:] = self.o_id
        attention_mask[-length_diff:] = 0

        #         print(tokenizer.convert_ids_to_tokens(input_ids))
        #         print(tags)
        #         print(attention_mask)

        #         print("Smaller tag filled in")
        #         raise Exception("FINISH")

        return input_ids, tags, attention_mask

    def augment_with_bigger_label(self, input_ids, tags, attention_mask, new_dataset, mention_positions, length_diff,
                                  new_dataset_len):
        # New token positions
        new_token_positions = torch.arange(mention_positions[0], mention_positions[0] + new_dataset_len)

        # Roll sentence away
        input_ids[new_token_positions[-1] + 1:] = input_ids[mention_positions[-1] + 1:length_diff].clone()
        tags[new_token_positions[-1] + 1:] = tags[mention_positions[-1] + 1:length_diff].clone()
        attention_mask[new_token_positions[-1] + 1:] = attention_mask[mention_positions[-1] + 1:length_diff].clone()

        # Set new tokens
        input_ids[new_token_positions] = new_dataset

        # Continue with I token!
        tags[mention_positions[-1] + 1:mention_positions[-1] + 1 - length_diff] = self.i_id
        attention_mask[mention_positions[-1] + 1:mention_positions[-1] + 1 - length_diff] = 1

        #         print(tokenizer.convert_ids_to_tokens(input_ids))
        #         print(tags)
        #         print(attention_mask)

        #         print("Bigger tag filled in")
        #         raise Exception("FINISH")

        return input_ids, tags, attention_mask


"""
Example usage:

train_dataset = BERTDataset(tr_inputs, tr_tags, tr_masks, tr_mention_pos, auxiliary_datasets, transform=MentionSwapAugment(0.5, o_id=tag2id[o_id], i_id=tag2id[i_id]))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
"""
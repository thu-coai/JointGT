import os
import numpy as np
import torch
import random

from transformers import BartTokenizer, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from modeling_bart import MyBartForConditionalGeneration as MyBart
from modeling_t5 import MyT5ForConditionalGeneration as MyT5
from modeling_bart import MyBartPretrain
from modeling_t5 import MyT5Pretrain
from data import WikidataDataset, WikidataDataLoader, WebNLGDataLoader, WebNLGDataset
from data import evaluate_bleu
from tqdm import tqdm, trange
import json


def run(args, logger):
    # Initialize tokenizer
    if args.model_name == "bart":
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

    if args.do_pretrain:
        # Pretrain on kgtext
        with open(args.knowledge_file + '.json', 'r') as f:
            kg_data = json.load(f)
        train_dataset = WikidataDataset(logger, args, args.train_file, kg_data, tokenizer, "train")
        dev_dataset = WikidataDataset(logger, args, args.predict_file, kg_data, tokenizer, "val")
        train_dataloader = WikidataDataLoader(args, train_dataset, "train")
        dev_dataloader = WikidataDataLoader(args, dev_dataset, "dev")
    else:
        # Finetune on webnlg / webquestions / pathquestions
        train_dataset = WebNLGDataset(logger, args, args.train_file, tokenizer, "train")
        dev_dataset = WebNLGDataset(logger, args, args.predict_file, tokenizer, "val")
        train_dataloader = WebNLGDataLoader(args, train_dataset, "train")
        dev_dataloader = WebNLGDataLoader(args, dev_dataset, "dev")

    if args.do_train:
        # Load model parameters
        if not args.do_pretrain:
            model = MyBart.from_pretrained(args.model_path) if args.model_name == "bart" \
                else MyT5.from_pretrained(args.model_path)
        else:
            model = MyBartPretrain.from_pretrained(args.model_path) if args.model_name == "bart" \
                else MyT5Pretrain.from_pretrained(args.model_path)

        print('model parameters: ', model.num_parameters())

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if not args.no_lr_decay:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=1000000)

        if not args.do_pretrain:
            train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer)
        else:
            pretrain(args, logger, model, train_dataloader, optimizer, scheduler)

    if args.do_predict:
        # Inference on the test set
        checkpoint = args.output_dir
        model = MyBart.from_pretrained(checkpoint) if args.model_name == "bart" \
            else MyT5.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=True)
        logger.info("%s on %s data: %.4f" % (dev_dataloader.dataset.metric, dev_dataloader.dataset.data_type, ems))


def pretrain(args, logger, model, train_dataloader, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    train_enc_loss, train_dec_loss, train_ot_loss = [], [], []
    task_ratio = eval(args.task_ratio)

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting pretraining!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            task_prob = random.random()
            if global_step == 1:
                for tmp_id in range(28):
                    print(batch[tmp_id])
            # Conduct the three subtasks with the probability in task_ratio
            if task_prob < task_ratio[0]:
                # complete graph + masked text (ar)
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                             decoder_input_ids=batch[2], decoder_attention_mask=batch[3], input_node_ids=batch[4],
                             input_edge_ids=batch[5], node_length=batch[6], edge_length=batch[7], adj_matrix=batch[8],
                             is_training=True)
            else:
                if task_prob < task_ratio[0] + task_ratio[1]:
                    # masked graph + complete text (ae)
                    loss = model(input_ids=batch[9], attention_mask=batch[10], encoder_label=batch[11],
                                 decoder_input_ids=batch[2], decoder_attention_mask=batch[3], input_node_ids=batch[12],
                                 input_edge_ids=batch[13], node_length=batch[14], edge_length=batch[15],
                                 adj_matrix=batch[16], is_training=True)
                else:
                    # complete graph + complete text (ot)
                    loss = model(input_ids=batch[17], attention_mask=batch[18],
                                 decoder_input_ids=batch[19], decoder_attention_mask=batch[20],
                                 decoder_whole_ids=batch[21], input_node_ids=batch[22], input_edge_ids=batch[23],
                                 node_length=batch[24], edge_length=batch[25], word_length=batch[26],
                                 adj_matrix=batch[27], is_training=True)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                break

            # Record training loss
            train_losses.append(loss.detach().cpu())
            if task_prob < task_ratio[0]:
                train_dec_loss.append(loss.detach().cpu())
            else:
                if task_prob < task_ratio[0] + task_ratio[1]:
                    train_enc_loss.append(loss.detach().cpu())
                else:
                    train_ot_loss.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            # Print loss
            if global_step % args.eval_period == 0:
                enc_loss_res = np.mean(train_enc_loss) if len(train_enc_loss) > 0 else 0.0
                dec_loss_res = np.mean(train_dec_loss) if len(train_dec_loss) > 0 else 0.0
                ot_loss_res = np.mean(train_ot_loss) if len(train_ot_loss) > 0 else 0.0
                logger.info("Step %d Encoder loss %.2f Decoder loss %.2f OT loss %.2f Learning rate %.2e epoch=%d" % (
                    global_step,
                    enc_loss_res,
                    dec_loss_res,
                    ot_loss_res,
                    scheduler.get_lr()[0],
                    epoch))
                train_losses = []
                train_enc_loss, train_dec_loss, train_ot_loss = [], [], []

            # Save model
            if global_step % args.save_period == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(args.output_dir)
                logger.info("Saving model on epoch=%d, global_step=%d" % (epoch, global_step))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        logger.info("Saving model on epoch=%d, global_step=%d" % (epoch, global_step))


def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if global_step == 1:
                for tmp_id in range(9):
                    print(batch[tmp_id])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],input_node_ids=batch[4],
                         input_edge_ids=batch[5], node_length=batch[6], edge_length=batch[7], adj_matrix=batch[8],
                         is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Print loss and evaluate on the valid set
            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_dataloader, tokenizer, args, logger)
                logger.info("Step %d Train loss %.2f Learning rate %.2e %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    dev_dataloader.dataset.metric,
                    curr_em * 100,
                    epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=False):
    predictions = []
    # Inference on the test set
    for i, batch in enumerate(dev_dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 input_node_ids=batch[4],
                                 input_edge_ids=batch[5],
                                 node_length=batch[6],
                                 edge_length=batch[7],
                                 adj_matrix=batch[8],
                                 num_beams=args.num_beams,
                                 length_penalty=args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True,)
        # Convert ids to tokens
        for input_, output in zip(batch[0], outputs):
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            predictions.append(pred.strip())

    # Save the generated results
    if save_predictions:
        save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            for pred in predictions:
                f.write(pred + '\n')
        logger.info("Saved prediction in {}".format(save_path))

    data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
    assert len(predictions) == len(data_ref)
    return evaluate_bleu(data_ref=data_ref, data_sys=predictions)

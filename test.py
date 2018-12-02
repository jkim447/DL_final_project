def train(args, model, optimizer, train_loader, epoch, total_minibatch_count,
                train_losses, train_accs, train_topk_accs):
        # Training for a full epoch
            
                model.train()
                    correct_count, total_loss, total_acc = 0., 0., 0.
                        progress_bar = tqdm.tqdm(train_loader, desc='Training')
                            
                                for batch_idx, (data, target) in enumerate(progress_bar):
                                            if args.cuda:
                                                            data, target = data.cuda(), target.cuda()
                                                                    data, target = Variable(data), Variable(target)

                                                                            # zero-out the gradients
                                                                                    optimizer.zero_grad()

                                                                                            # Forward prediction step
                                                                                                    output = model(data)

                                                                                                            # find the loss
                                                                                                                    loss = F.nll_loss(output, target)

                                                                                                                            # do backprop
                                                                                                                                    loss.backward()
                                                                                                                                            optimizer.step()
                                                                                                                                                    
                                                                                                                                                            # The batch has ended, determine the accuracy of the predicted outputs
                                                                                                                                                                    pred = output.data.max(1)[1]  
                                                                                                                                                                            
                                                                                                                                                                                    # target labels and predictions are categorical values from 0 to 9.
                                                                                                                                                                                            matches = target == pred
                                                                                                                                                                                                    accuracy = matches.float().mean()
                                                                                                                                                                                                            correct_count += matches.sum()

                                                                                                                                                                                                                    total_loss += loss.data
                                                                                                                                                                                                                            total_acc += accuracy.data
                                                                                                                                                                                                                                    progress_bar.set_description(
                                                                                                                                                                                                                                                        'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                                                                                                                                                                                                                                                                            epoch, total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)))
                                                                                                                                                                                                                                                                #progress_bar.refresh()

                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                # log data every interval or so
                                                                                                                                                                                                                                                                                        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:

                                                                                                                                                                                                                                                                                                        train_losses.append(loss.data[0].cpu().numpy())
                                                                                                                                                                                                                                                                                                                    train_accs.append(accuracy.data[0].cpu().numpy())
                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                            # calculate topk accuracy
                                                                                                                                                                                                                                                                                                                                                        batch_size=target.size(0)
                                                                                                                                                                                                                                                                                                                                                                    _, pred_topk = output.topk(5,1,True,sorted=True)
                                                                                                                                                                                                                                                                                                                                                                                pred_topk = pred_topk.t()
                                                                                                                                                                                                                                                                                                                                                                                            correct_topk=pred_topk.eq(target.view(1,-1).expand_as(pred_topk))
                                                                                                                                                                                                                                                                                                                                                                                                        correct_topk = correct_topk[:5].view(-1).float().sum(0,keepdim=True)
                                                                                                                                                                                                                                                                                                                                                                                                                    correct_topk = correct_topk.mul_(100.0/batch_size)
                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                train_topk_accs.append(correct_topk.data[0].cpu().numpy())
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # write to csv file
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    print("logging csv now")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                with open(os.path.join(os.getcwd(),'train.csv'), 'w') as f:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    csvw = csv.writer(f, delimiter=',')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    for loss, acc,topk_accs in zip(train_losses, train_accs, train_topk_accs):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            csvw.writerow((loss, acc, topk_accs))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                total_minibatch_count += 1

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    return total_minibatch_count

import torch.nn as nn
import torch 
import numpy as np
import matplotlib.pyplot as plt
from models.extras import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following fucntions are modified versions of traning loops, eahc using a different techniques
# All parameters wioll be described here to avoid repetition
# Each function will have more details described in the docstring 
# Parameters:
# ----------
# model - The neural network model to be trained
# optimizer - The optimizer used to update model parameters
# scheduler - The learning rate scheduler
# n_epochs - The number of training epochs
# train_loader - DataLoader for the training dataset
# val_loader - DataLoader for the validation dataset
# min_beta - Minimum value for the beta parameter in the cyclic annealing schedule
# max_beta - Maximum value for the beta parameter in the cyclic annealing schedule
# gamma_gene_abundance_start - Initial value of the scaling factor for the gene abundance loss
# gamma_gene_abundance_end - Final value of the scaling factor for the gene abundance loss
# gamma_genome_size_start - Initial value of the scaling factor for the genome size loss
# gamma_genome_size_end - Final value of the scaling factor for the genome size loss
# max_norm - Maximum norm for gradient clipping
# lambda_l1 - Regularization strength parameter for L1 regularization

# Returns:
# -------
# train_loss_vals - List of average training losses for each epoch
# val_loss_vals - List of average validation losses for each epoch
# epoch + 1 - The number of completed epochs, including early stopping if triggered
# and the graphs that were produced during the training as pdf files 

def v3(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, min_beta, max_beta, gamma_start, gamma_end, weight, max_norm, lambda_l1):
    '''
    Training function for a VAE model which uses linear KL annealing, gradient clipping, early stoppping, l1 regularisation and modified loss fucntion wich includes gene_abundance AND genome size which use linear annealing wiht different coefficients 

    '''

    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    T = 50
    counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20
    min_delta = 1e-4

    train_recon_loss_vals = []
    train_kl_loss_vals = []
    train_gene_loss_vals = []
    train_l1_loss_vals = []

    val_recon_loss_vals = []
    val_kl_loss_vals = []
    val_gene_loss_vals = []

    for epoch in range(n_epochs):
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        model.train()

        epoch_train_loss = 0.0
        epoch_train_recon_loss = 0.0
        epoch_train_kl_loss = 0.0
        epoch_train_gene_loss = 0.0
        epoch_train_l1_loss = 0.0

        for batch in train_loader:
            t = epoch * 32 + counter
            beta = cosine_annealing_schedule(t, T, min_beta, max_beta)
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)

            total_gene_number = recon_x.sum(axis=0)
            total_gene_number_loss = torch.sum(torch.abs(total_gene_number))

            l1_penalty = l1_regularization(model, lambda_l1)

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Use original losses for backpropagation
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (weight * gamma * total_gene_number_loss) + l1_penalty 

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            counter += 1

            scaled_reconstruction_loss = reconstruction_loss / loss
            scaled_kl_divergence_loss = (beta * kl_divergence_loss) / loss
            scaled_gene_loss = (weight * gamma * total_gene_number_loss) / loss
            scaled_l1_penalty = l1_penalty / loss

            epoch_train_loss += loss.item()
            epoch_train_recon_loss += scaled_reconstruction_loss.item()
            epoch_train_kl_loss += scaled_kl_divergence_loss.item()
            epoch_train_gene_loss += scaled_gene_loss.item()
            epoch_train_l1_loss += scaled_l1_penalty.item()

        train_recon_loss_vals.append(epoch_train_recon_loss / len(train_loader.dataset))
        train_kl_loss_vals.append(epoch_train_kl_loss / len(train_loader.dataset))
        train_gene_loss_vals.append(epoch_train_gene_loss / len(train_loader.dataset))
        train_l1_loss_vals.append(epoch_train_l1_loss / len(train_loader.dataset))

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_recon_loss = 0.0
        epoch_val_kl_loss = 0.0
        epoch_val_gene_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                total_gene_number = recon_x.sum(axis=0)
                total_gene_number_loss = torch.sum(torch.abs(total_gene_number))

                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                loss = reconstruction_loss + (beta * kl_divergence_loss) + (weight * gamma * total_gene_number_loss)
                
                scaled_reconstruction_loss = reconstruction_loss / loss
                scaled_kl_divergence_loss = (beta * kl_divergence_loss) / loss
                scaled_gene_loss = (weight * gamma * total_gene_number_loss) / loss

                epoch_val_loss += loss.item()
                epoch_val_recon_loss += scaled_reconstruction_loss.item()
                epoch_val_kl_loss += scaled_kl_divergence_loss.item()
                epoch_val_gene_loss += scaled_gene_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        val_recon_loss_vals.append(epoch_val_recon_loss / len(val_loader.dataset))
        val_kl_loss_vals.append(epoch_val_kl_loss / len(val_loader.dataset))
        val_gene_loss_vals.append(epoch_val_gene_loss / len(val_loader.dataset))

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

        # Print some example values of each loss to diagnose the issue
    print("Example values for losses:")
    print("Train Reconstruction Loss:", train_recon_loss_vals[:10])
    print("Train KL Loss:", train_kl_loss_vals[:10])
    print("Train Gene Abundance Loss:", train_gene_loss_vals[:10])
    print("Train L1 Loss:", train_l1_loss_vals[:10])

    # Plot each loss separately
    plt.figure(figsize=(12, 4), dpi=300)

    plt.subplot(3, 2, 1)
    plt.plot(train_recon_loss_vals, label='Train Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.subplot(3, 2, 2)
    plt.plot(train_kl_loss_vals, label='Train KL Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.subplot(3, 2, 3)
    plt.plot(train_gene_loss_vals, label='Train Gene Abundance Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.subplot(3, 2, 5)
    plt.plot(train_l1_loss_vals, label='Train L1 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(folder+f"train_losses_separated_{weight}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Log scale plot
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(train_recon_loss_vals, label='Train Reconstruction Loss')
    plt.plot(train_kl_loss_vals, label='Train KL Loss')
    plt.plot(train_gene_loss_vals, label='Train Gene Abundance Loss')
    plt.plot(train_l1_loss_vals, label='Train L1 Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend(fontsize=8)
    plt.savefig(folder+f"train_losses_log_scale_{weight}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plotting training loss components
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(train_recon_loss_vals, label='Train Reconstruction Loss')
    plt.plot(train_kl_loss_vals, label='Train KL Loss')
    plt.plot(train_gene_loss_vals, label='Train Gene Abundance Loss')
    plt.plot(train_l1_loss_vals, label='Train L1 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)
    plt.savefig(folder+f"train_losses_{weight}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Plotting validation loss components
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(val_recon_loss_vals, label='Val Reconstruction Loss')
    plt.plot(val_kl_loss_vals, label='Val KL Loss')
    plt.plot(val_gene_loss_vals, label='Val Gene Abundance Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)
    plt.savefig(folder+f"validation_losses_{weight}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return train_loss_vals, val_loss_vals, epoch + 1

def v2(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, min_beta, max_beta, gamma_start, gamma_end, max_norm, lambda_l1):
    '''
    Training function for a VAE model which uses cyclic KL annealing, gradient clipping, early stoppping, l1 regularisation and modified loss fucntion wich includes gene_abundance which follows linear annealing (technically absolute value of all genes in the dataset)

    '''

    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    T = 10
    counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4

    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        # beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            t = epoch * 32 + counter
            beta = cosine_annealing_schedule(t, T, min_beta, max_beta)
            # gamma = exponential_decay_schedule(t, gamma_start, decay_rate)
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)


            total_gene_number = recon_x.sum(axis=0)
            total_gene_number_loss = torch.sum(torch.abs(total_gene_number)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * total_gene_number_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            counter += 1

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                total_gene_number = recon_x.sum(axis=0)
                total_gene_number_loss = torch.sum(torch.abs(total_gene_number)) 

                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * total_gene_number_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals, epoch + 1

def v1(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, beta_start, beta_end, gamma_start, gamma_end, max_norm, lambda_l1):
    '''
    Training function for a VAE model which uses linear KL annealing, gradient clipping, early stoppping, l1 regularisation and modified loss fucntion wich includes gene_abundance (technically absolute value of all genes in the dataset)

    '''

    train_loss_vals = []
    val_loss_vals = []
    train_loss = 0.0
    val_loss = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4
    #initial_scaling_factor = 1e-3

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        gamma = gamma_start + (gamma_end - gamma_start) * epoch / n_epochs
        #scaling_factor = initial_scaling_factor / (epoch + 1)
        model.train()

        epoch_train_loss = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)

            total_gene_number = recon_x.sum(axis=0)
            total_gene_number_loss = torch.sum(torch.abs(total_gene_number)) 

            l1_penalty = l1_regularization(model, lambda_l1)

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * total_gene_number_loss) + l1_penalty

            loss.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        train_loss_vals.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)

                total_gene_number = recon_x.sum(axis=0)
                total_gene_number_loss = torch.sum(torch.abs(total_gene_number)) 

                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + (beta * kl_divergence_loss) + (gamma * total_gene_number_loss)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                  f" Train Loss: {avg_train_loss}\n"
                  f" Validation Loss: {avg_val_loss}")

        train_loss += avg_train_loss
        val_loss += avg_val_loss

        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    final_avg_train_loss = train_loss / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    print(f"Final Average Training Loss: {final_avg_train_loss}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals, val_loss_vals, epoch + 1

def v0(model, folder, optimizer, scheduler, n_epochs, train_loader, val_loader, beta_start, beta_end, max_norm):
    '''
    basic trining function for VAE with linear KL annealing, gradient clipping  and early stoppping implemented 

    '''

    # global train_loss_vals 
    # train_loss_vals = []
    # global train_loss_vals2 
    train_loss_vals2 = []
    # global val_loss_vals
    val_loss_vals = []
    # train_loss = 0.0
    train_loss2 = 0.0
    val_loss = 0.0
    # best_val_loss = float('inf')
    # early_stopping_patience = 5
    # early_stopping_counter = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience=10
    min_delta=1e-4

    for epoch in range(n_epochs):
        beta = beta_start + (beta_end - beta_start) * epoch / n_epochs
        model.train()

        # epoch_train_loss = 0.0
        epoch_train_loss2 = 0.0

        for batch in train_loader:
            data = batch[0].to(torch.float).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)
            # print('reco_x:', recon_x[:1, :5])
            # print('data:', data[:1, :5])

            # print(recon_x.shape)
            # print(data.shape) 

            reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
            kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # loss = reconstruction_loss + kl_divergence_loss
            loss2 = reconstruction_loss + (beta * kl_divergence_loss)

            loss2.backward()
            
            # Need to read more on gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            # epoch_train_loss += loss.item()
            epoch_train_loss2 += loss2.item()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} gradient: {param.grad.abs().mean().item()}') 

        # avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_train_loss2 = epoch_train_loss2 / len(train_loader.dataset)
        # train_loss_vals.append(avg_train_loss)
        train_loss_vals2.append(avg_train_loss2)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(torch.float).to(device)
                recon_x, mu, logvar = model(data)
                reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, data, reduction='sum')
                kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss2 = reconstruction_loss + (beta * kl_divergence_loss)

                epoch_val_loss += loss2.item()

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_loss_vals.append(avg_val_loss)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:\n"
                  f" Learning Rate: {scheduler.get_last_lr()[0]}\n"
                #   f" Train Loss (method 1): {avg_train_loss}\n"
                  f" Train Loss (method 2): {avg_train_loss2}\n"
                  f" Validation Loss: {avg_val_loss}")

        # train_loss += avg_train_loss
        train_loss2 += avg_train_loss2
        val_loss += avg_val_loss

        # # Check for early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     early_stopping_counter = 0
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter >= early_stopping_patience:
        #     print("Early stopping triggered")
        #     break


        #  Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # final_avg_train_loss = train_loss / n_epochs
    final_avg_train_loss2 = train_loss2 / n_epochs
    final_avg_val_loss = val_loss / n_epochs

    # print(f"\nFinal Average Training Loss (method 1): {final_avg_train_loss}")
    print(f"Final Average Training Loss (method 2): {final_avg_train_loss2}")
    print(f"Final Average Validation Loss: {final_avg_val_loss}")

    return train_loss_vals2, val_loss_vals, epoch + 1

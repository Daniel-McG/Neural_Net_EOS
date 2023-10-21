import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import ray.train.lightning
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
import pickle
import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search import Repeater
from ray.train.torch import TorchConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune import CLIReporter
reporter = CLIReporter(max_progress_rows=10)
logger = TensorBoardLogger("tb_logs", name="my_model")
ray.init()
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
#Defining the neural 

class BasicLightning(pl.LightningModule):
    def __init__(self,config):
        super(BasicLightning,self).__init__() 
        self.l1_size = config["layer_1_size"]
        self.l2_size = config["layer_2_size"]
        self.l3_size = config["layer_3_size"]
        self.l4_size = config["layer_4_size"]
        self.l5_size = config["layer_5_size"]
        self.l6_size = config["layer_6_size"]
        self.l7_size = config["layer_7_size"]
        self.lr = config["lr"]
        # Creating a sequential stack of Linear layers with Tanh activation function 

        self.s1 = nn.Sequential(
          nn.Linear(2,self.l1_size),
          nn.Tanh(),
          nn.Linear(self.l1_size,self.l2_size),
          nn.Tanh(),
          nn.Linear(self.l2_size,self.l3_size ),
          nn.Tanh(),
          nn.Linear(self.l3_size ,self.l4_size ),
          nn.Tanh(),
          nn.Linear(self.l4_size ,self.l5_size),
          nn.Tanh(),
          nn.Linear(self.l5_size,self.l6_size),
          nn.Tanh(),
          nn.Linear(self.l6_size,self.l7_size),
          nn.Tanh(),
          nn.Linear(self.l7_size,2),
          nn.Tanh()
        )

    def forward(self,x):
        out = self.s1(x)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = self.lr)
    
    def training_step(self,train_batch,batch_index):
        input_i,target_i = train_batch        #Unpacking data from a batch
        output_i = self.forward(input_i)    #Putting input data frm the batch through the neural network
        loss = (output_i-target_i)**2       #Calculating loss
        mean_loss = torch.mean(loss)
        self.log("train_loss",mean_loss)         #Logging the training loss
        return {"loss": mean_loss}
    
    def validation_step(self, val_batch, batch_idx):
        val_input_i, val_target_i = val_batch
        val_output_i = self.forward(val_input_i)
        loss = (val_output_i-val_target_i)**2
        mean_loss = torch.mean(loss)
        self.log("val_loss",mean_loss) 
        return {"val_loss": mean_loss}

def train_func(config):
    # Read data from csv
    data_df = pd.read_csv('/home/daniel/Downloads/MSc_data.csv',names=['rho','T','P','U'])

    #Preprocessing the data
    train_df,test_df = train_test_split(data_df,train_size=0.6)
    scaler = MinMaxScaler(feature_range =(0,1))
    train_arr= scaler.fit_transform(train_df)
    val_arr = scaler.transform(test_df)
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    #Plotting distribution of train and test data
    # for column in range(0,4,1):
    #     plt.clf()
    #     train_histplot = sns.histplot(data = train_arr[:,column])
    #     train_fig = train_histplot.get_figure()
    #     train_fig.savefig("/home/daniel/Pictures/BS_32_train_col{}.png".format(str(column)))

    # for column in range(0,4,1):
    #     plt.clf()
    #     val_histplot = sns.histplot(data = val_arr[:,column])
    #     val_fig = val_histplot.get_figure()
    #     val_fig.savefig("/home/daniel/Pictures/BS_32_val_col{}.png".format(str(column)))

    #Splitting the preprocessed data into the inputs and targets
    train_inputs = torch.tensor(train_arr[:,[0,1]])
    train_targets = torch.tensor(train_arr[:,[2,3]])
    val_inputs = torch.tensor(val_arr[:,[0,1]])
    val_targets = torch.tensor(val_arr[:,[2,3]])
    train_inputs = train_inputs.float()
    train_targets = train_targets.float()
    val_inputs = val_inputs.float()
    val_targets = val_targets.float()

    # Loading inputs and targets into the dataloaders
    train_dataset = TensorDataset(train_inputs,train_targets)
    val_Dataset = TensorDataset(val_inputs,val_targets)
    train_dataloader = DataLoader(train_dataset,batch_size = config["batch_size"])
    val_dataloader = DataLoader(val_Dataset,batch_size = config["batch_size"])
    model = BasicLightning(config)

    trainer = pl.Trainer(
        max_epochs=20000,
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback(),EarlyStopping(monitor="val_loss",mode="min",patience=500)],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


scaling_config = ScalingConfig(num_workers=1, use_gpu=False)
run_config = RunConfig(progress_reporter=reporter,
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
)

trainer = TorchTrainer(train_func, scaling_config=scaling_config,run_config=run_config,)


# Tuning
uniform_dist = tune.randint(32,250)
search_space = {
    "layer_1_size": uniform_dist,
    "layer_2_size": uniform_dist,
    "layer_3_size": uniform_dist,
    "layer_4_size": uniform_dist,
    "layer_5_size": uniform_dist,
    "layer_6_size": uniform_dist,
    "layer_7_size": uniform_dist,
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([32,64,128,128*2])
}

num_samples = 10000


def tune_mnist_asha(num_samples=num_samples):
    scheduler = ASHAScheduler(max_t= 40000 , grace_period=100, reduction_factor=2)
    algo = NevergradSearch(
    optimizer=ng.optimizers.PSO)
    tuner = tune.Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=num_samples)
results.get_best_result(metric="val_loss", mode="min")
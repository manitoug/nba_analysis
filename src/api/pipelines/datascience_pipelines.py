from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.svm import LinearSVC
from typing import Literal, Any
import pandas as pd


@dataclass
class DataSciencePipeline(ABC):

    features: list[str]
    target: str
    model: Any = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    input_file: str | None = None
    
    

    @abstractmethod
    def load_data(self):
        ...

    @abstractmethod
    def fit(self):          
        ...

    @abstractmethod
    def predict(self):
        ...

@dataclass
class LinearSVCPipeline(DataSciencePipeline):

    C: int = 1  
    loss: Literal['hinge', 'squared_hinge'] = "hinge"
    class_weight: dict[int,int] | Literal['balanced'] = "balanced"
    max_iter: int =100000

    def load_data(self):
         data = pd.read_csv(self.input_file, engine="python")
         self.X = data.loc[:, self.features]
         self.y = data.loc[:, self.target].astype(int)
    
    def create_model(self):
        self.model = LinearSVC(C= self.C, loss=self.loss, 
                               class_weight=self.class_weight,
                               max_iter=self.max_iter)
    
    def fit(self):
        self.model.fit(self.X, self.y)  

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
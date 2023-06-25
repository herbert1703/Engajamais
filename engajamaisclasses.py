# -*- coding: utf-8 -*-
"""Engajamais fases 1_2_3_4.ipynb
"""
#######################################################################################################################
# Instalar e importar Pacotes Necessários
#######################################################################################################################
import subprocess
def pipinstall(name):
    subprocess.call(['pip', 'install', name])

# Instalação de pacotes

print('Instalando versão 1.1.3 do scikit-learn')
pipinstall('scikit-learn==1.1.3')
print('Instalando Pacote Shap...')
pipinstall('shap')
print('Instalando Pacote catboost...')
pipinstall('catboost')
print('Instalando Pacote BorutaShap...')
pipinstall('BorutaShap')
# Pacote para verificar outlier com a biblioteca PyOD
print('Instalando Pacote Pyod para detecção de outlier...')
pipinstall('pyod')

# Importação de Pacotes
import pandas as pd
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as ltb
import pickle
import scipy.stats
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.preprocessing import MinMaxScaler,RobustScaler,OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
from scipy.stats import describe,loguniform,wilcoxon
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import cross_val_score,cross_validate,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV,KFold,StratifiedKFold
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from BorutaShap import BorutaShap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from IPython.display import display, HTML
import math
import copy

pd.set_option('max_colwidth', -1)

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.weight': 'bold', 'font.size': 10})

#######################################################################################################################
# Funções de apoio (utils) para as classes
#######################################################################################################################
"""
**Funções para Split e configuração de scores**
"""
def executargridsearch(model,params,X_,y_,scores='f1_macro',silent='N'):
  folds = config_experimento.num_folds
  param_comb = 10
  skf = definesplit(num_folds=folds,rand_state=42)
  #StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
  random_search = RandomizedSearchCV(model,
                                     param_distributions=params,
                                     n_iter=param_comb,
                                     scoring='f1_macro',
                                     n_jobs=4,
                                     cv=skf.split(X_,y_),
                                     verbose=3,
                                     random_state=42)

  random_search.fit(X_, y_)
  if silent == 'N':
    print('   Best normalized gini score for %d-fold search:' % 
          (folds),random_search.best_score_)
    print('   Best hyperparameters:',random_search.best_params_)
  results = pd.DataFrame(random_search.cv_results_)

  return random_search

def definesplit(num_folds,rand_state,Predict=False):
  # Mais apropriado para dados desbalanceados
  if (config_experimento.Tiposplit == 2) or (Predict):
    split = StratifiedKFold(n_splits=num_folds, shuffle=True, 
                            random_state=rand_state)
  else:
    split =  RepeatedStratifiedKFold(n_splits=num_folds,
                                     n_repeats=config_experimento.n_repeats, random_state=42)
  return split

def evaluate_model(model, X, y, scores, n_splits=5, n_repeats=30):
    cv = RepeatedStratifiedKFold(n_splits=n_splits,
                                 n_repeats=n_repeats, random_state=42)

    scores_result = cross_validate(model, X, y, scoring=scores,
                             cv=cv, n_jobs=-1, error_score='raise',
                             return_estimator=True)

    return scores_result

def definedf_result():
    names = []
    for modelo in config_experimento.modelos:
      names.append(modelo['nome_classificador'])
    nrows = 2
    # (config_experimento.num_folds * config_experimento.n_repeats)
    ncols =  len(names)
    results = pd.DataFrame(np.zeros((nrows,ncols)), columns=names)
    iterables = [list(names), config_experimento.scores]
    mult_idx = pd.MultiIndex.from_product(iterables, names=["CLF", "SCORE"])
    nrows = len(mult_idx)
    df_results = pd.DataFrame(np.zeros((nrows,2)),
                              index=mult_idx)
    return df_results

def set_scores_result(df_results, scores_result, model_name):
  vmetfresult= ['accuracy','precision','recall','f1','roc_auc']
  vmetcross  = ['vacuracia','vprecisao','vrecall','vf1','vroc']
  vmetteste  = ['vaccmodelkfold','vprecmodelkfold','vrecallmodelkfold',
                'vf1modelkfold','vrocaucmodelkfold']

  for index in range(len(vmetfresult)):
    df_results.loc[model_name, 
      vmetfresult[index]][0]=np.median(scores_result[vmetcross[index]])
    df_results.loc[model_name, 
      vmetfresult[index]][1]=np.median(scores_result[vmetteste[index]])

"""**Funções para gerar gráficos**"""
def matriz_confusao(y_test, y_predict,modelo,nome):
    matriz_conf = confusion_matrix(y_test, y_predict)
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(matriz_conf, annot=True, cmap='Blues', ax=ax,fmt='g');
    ax.set_xlabel('Valor Predito');
    ax.set_ylabel('Valor Real');
    ax.set_title('Matriz de Confusão - '+nome);
    #ax.xaxis.set_ticklabels(modelo.classes_);
    #ax.yaxis.set_ticklabels(modelo.classes_);
    ax.xaxis.set_ticklabels([0,1]);
    ax.yaxis.set_ticklabels([0,1]);
    #plt.close()
    plt.show()
    return fig

def plota_heatmap(X_):
  plt.figure(figsize=(20,20))
  plt.rcParams.update({'font.family' :'sans-serif',
                              'font.weight': 'bold','font.size': 10})
  sns.heatmap(X_.corr(method='spearman'),vmin=-1,vmax=1,fmt='.2g',annot=True,cmap='YlGnBu')

def montargraficoevasao(dfnumevadidos,x_='curso',nomecurso=''):
  font = {'family' : 'serif',
              'color'  : 'darkred',
              'weight' : 'bold',
              'size'   : 20,
              }
  plt.figure(figsize=(50, 6))
  plt.rcParams.update({'font.family' :'sans-serif',
                              'font.weight': 'bold','font.size': 20})

  ax = dfnumevadidos.plot(x=x_,y=['evadidos','totalturma'],kind = 'bar',figsize=(20, 10),
                    fontsize=15,rot=15,color={'evadidos': '#ff0051','totalturma':'#008bfb'})

  plt.ylabel("Quantidade de alunos",fontdict=font)
  plt.xlabel(x_,fontdict=font)
  x_offset = -0.08
  y_offset = 0.50
  for p in ax.patches:
    b = p.get_bbox()
    val = "{:.0f}".format(b.y1 + b.y0)
    ax.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
  plt.savefig("graficoevasao.pdf", format='pdf',dpi=300, bbox_inches = 'tight')
  plt.show()
  plt.close()

def graficoevasao_periodo(df_,nomevar_y = 'situacao'):
  # Padrão de evasão por período
  font = {'family' : 'serif',
                'color'  : 'darkred',
                'weight' : 'normal',
                'size'   : 15,
                }
  evasao_periodo = df_[(df_[config_experimento.var_y] == 1)].groupby(
      ['periodo_modulo',config_experimento.var_y],
      as_index=False)['curso'].count()
  bottom, height = .25, .5
  plt.figure(figsize=(50, 6))
  ax = evasao_periodo.plot(x='periodo_modulo',y=['curso'],kind = 'bar',
                           figsize=(20, 10),
                           fontsize=15,rot=15,color={'curso':'royalblue'})
  plt.title("Evasão por Período_módulo",fontdict=font)

  plt.ylabel("Qtde Evadidos",fontdict=font)
  plt.xlabel("Período_Módulo",fontdict=font)
  for p in ax.patches:
          txt = "{:.2f}".format(p.get_height().round(1)/evasao_periodo['curso'].sum()*100) + '%'
          txt_x = p.get_x()
          txt_y = p.get_height()
          x_offset = 0
          y_offset = 1
          b = p.get_bbox()
          ax.annotate(txt, ((b.x0) + x_offset, b.y1 + y_offset))
          #ax.text(txt_x,txt_y,txt)
  ax.legend(['% Evadidos'])
  plt.show()

def boxplot_sorted(modelos_,resultadosmodel, metric='f1_score', by=['Technique'],
                   rot=90, figsize=(18,8), fontsize=24, section='',df_definido=None):
    if df_definido is None:
        df2 = pd.DataFrame({modelos_[i]:resultadosmodel[i]
                                     [metric] for i in range(0,len(modelos_))})
    else:
        df2 = df_definido
    df2 = df2.round(decimals=4)
    meds = df2.median().sort_values(ascending=False)
    axes = df2[meds.index].boxplot(figsize=figsize, rot=rot, 
                                   fontsize=fontsize,
                                   boxprops=dict(linewidth=4, 
                                                 color='cornflowerblue'),
                                   whiskerprops=dict(linewidth=4, 
                                                     color='cornflowerblue'),
                                   medianprops=dict(linewidth=4, 
                                                    color='firebrick'),
                                   capprops=dict(linewidth=4, 
                                                 color='cornflowerblue'),
                                   flierprops=dict(marker='o',
                                        markersize=12,
                                         markeredgecolor='black'),
                                   return_type="axes")

    str_by = '_'.join([str(s) for s in by])
    str_title = 'Boxplots da metrica %s por %s'
    axes.set_title(str_title % (metric,str_by), fontsize=fontsize+4)

    # Save figure
    figname = str.lower(f'boxplot_{section}_top_{str_by}_{metric}' +'.pdf')
    plt.savefig(figname, format='pdf', bbox_inches = 'tight')
    plt.show()
    plt.close()

def plotresult(crossval, tests, metrica,section=''):
  plt.figure(figsize=(24,10))
  plt.clf()
  plt.plot(range(len(crossval)),crossval,
          label="Cross_validate por iteração com RepeatedStratifiedKFold")
  plt.plot(range(len(tests)),tests,
          label="Teste por iteração train_test_split (30%,shuffle e stratify em target)")
  plt.xlabel("iterações")
  plt.ylabel(metrica)
  plt.legend()
  figname = str.lower(f'Iteracoescrossval_{section}_{metrica}' +'.pdf')
  plt.savefig(figname, format='pdf', bbox_inches = 'tight')
  plt.show()
  plt.close()

def histplotX(X_):

  vnrows = math.ceil(len(X_.columns)/4)
  fig, axes = plt.subplots(nrows = vnrows, ncols = 4)
  axes = axes.flatten()
  fig.set_size_inches(20, 20)

  for ax, col in zip(axes, X_.columns):
    sns.histplot(X_[col], ax = ax,kde=True)

def plotadispersao(finalDf_):
  fig = plt.figure(figsize = (8,8))
  ax = fig.add_subplot(1,1,1)
  ax.set_xlabel('Componente Principal 1', fontsize = 15)
  ax.set_ylabel('Componente Principal 2', fontsize = 15)
  ax.set_title('Dois componentes PCA', fontsize = 20)
  targets = [0, 1]
  #colors = ['r', 'g', 'b']
  colors = ['r', 'g']
  for target, color in zip(targets,colors):
      indicesToKeep = finalDf_[config_experimento.var_y] == target
      ax.scatter(finalDf_.loc[indicesToKeep, 'principal_component_1']
                , finalDf_.loc[indicesToKeep, 'principal_component_2']
                , c = color
                , s = 50)
  ax.legend(targets)
  ax.grid()
  plt.show()

def plota_auc(y_testpred, previsao_mod):
  # Compute False postive rate, and True positive rate
  fpr, tpr, thresholds =roc_curve(y_testpred, previsao_mod)
  plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('', vauc))

  # Custom settings for the plot
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('1-Specificity(False Positive Rate)')
  plt.ylabel('Sensitivity(True Positive Rate)')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()   # Display

  # Compute False postive rate, and True positive rate
  fpr, trc, thresholds=precision_recall_curve(y_testpred, previsao_mod)
  vauc = auc(trc, fpr)
  print("AUC-PR ",vauc)
  plt.plot(fpr, trc, label='%s AUCPR (area = %0.2f)' % ('', vauc))

  # Custom settings for the plot
  no_skill = len(y_testpred[y_testpred==1]) / len(y_testpred)
  plt.plot([0, 1], [no_skill, no_skill],'r--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall (Positive Label: 1)')
  plt.ylabel('Precision (Positive Label: 1)')
  plt.title('Precision-Recall curve')
  plt.legend(loc="lower right")
  plt.show()   # Display

#######################################################################################################################
# Classe Singleton para configuração dos experimentos
#######################################################################################################################
class Configexperimento:

  def __init__(self):
    self.featuresselecionadas = None
    self.scores = ['accuracy','recall','f1','precision','roc_auc','recall_macro']
    self.modelos = []
    self.var_y = 'situacao'
    self.combalanceamento = True
    self.comnormalizacao = False
    self.Tiposplit = 1
    self.otimizahiperparametros = False
    self.num_folds=5
    self.n_repeats=3
    self.perc_train = 0.70
    self.perc_test = 0.30
    self.remove_outlier = True


  def adiciona_modelo(self,modelo):
    idx = [i for i, j in
           enumerate(self.modelos) if j['nome_classificador'] == modelo['nome_classificador']]
    if len(idx) > 0:
      self.modelos[idx[0]] = modelo
    else:
      self.modelos.append(modelo)

  def deleta_modelo(self,nomemodelo):
    idx = [i for i, j in
           enumerate(self.modelos) if j['nome_classificador'] == nomemodelo]
    if len(idx) > 0:
      self.modelos.pop(idx[0])

  def defval_cfg_modelo(self,nomemodelo,cfg,valor):
    idx = [i for i, j in
           enumerate(self.modelos) if j['nome_classificador'] == nomemodelo]
    if len(idx) > 0:
      self.modelos[idx[0]][cfg] = valor
    else:
      print('Nenhum Valor Alterado')

  def retval_cfg_modelo(self,nomemodelo,cfg):
    idx = [i for i, j in
           enumerate(self.modelos) if j['nome_classificador'] == nomemodelo]
    if len(idx) > 0:
      return self.modelos[idx[0]][cfg]
    else:
      return None

  def info(self):
    print("Features selecionadas:",self.featuresselecionadas)
    print("Métricas utilizadas..:",self.scores)
    print("Variável Alvo........:",self.var_y)
    print("Executa balanceamento:",self.combalanceamento)
    print("Normalização.........:",self.comnormalizacao)
    print("Remove outliers......:",self.remove_outlier)
    print("Utiliza otimiz Parâm.:",self.otimizahiperparametros)
    print("Tipo Split de dados .:",self.Tiposplit)
    print("N_repeats Stratified.:",self.n_repeats)
    print("K_folds Stratified...:",self.num_folds)
    print("Perc.Treinamento.....:",self.perc_train)
    print("Perc.Teste...........:",self.perc_test)
    print("Classificadores......:")
    for modelo in self.modelos:
      print(modelo)

config_experimento = Configexperimento()

#######################################################################################################################
# Classe para análise exploratória dos dados
#######################################################################################################################
class Engaja_analise_Expl:
  
  def __init__(self,nomevarevasao='situacao',
                    nomevarcurso='curso',nomevaraluno='aluno'):
    self.nomevarevasao = nomevarevasao
    self.nomevarcurso = nomevarcurso
    self.nomevaraluno = nomevaraluno

  def __calcula_perc(self,linha):
    return "{:.2f}".format(linha['evadidos']/linha['totalturma']*100)
  
  def __verificadados(self,df_,features_):
    if (df_ is None):
      raise Exception("DataFrame Não pode estar vazio")
    if ((features_ is None) or 
        not(type(features_) is list)):
      raise Exception("Lista de Features não pode estar vazia "+
                      "e precisa ser do tipo list")

  def plota_situacao_evasao(self,df_):
    self.__verificadados(df_,config_experimento.featuresselecionadas)

    dfnumevadidos_ = pd.DataFrame((df_.where(df_[self.nomevarevasao] == 1)).
                                groupby([self.nomevarcurso],
                                        as_index=False)[self.nomevaraluno].agg(['nunique']))
    dftotalturma_ = df_.groupby([self.nomevarcurso])[self.nomevaraluno].agg(['nunique'])
    vnumevadidos = []
    dftotalturma_ = dftotalturma_.reset_index()
    dftotalturma_ = dftotalturma_.rename(columns={'nunique':'totalturma'})
    dfnumevadidos_ = dfnumevadidos_.reset_index()
    dfnumevadidos_=dfnumevadidos_.rename(columns={'nunique':'evadidos'})
    for (i,row) in dftotalturma_.iterrows():
      vct = dfnumevadidos_[dfnumevadidos_[self.nomevarcurso] == row[self.nomevarcurso]].count()[0]
      if vct > 0:
        vnumevadidos.append(dfnumevadidos_[dfnumevadidos_[self.nomevarcurso] == 
                                          row[self.nomevarcurso]]['evadidos'].values[0])
      else:
        vnumevadidos.append(0)

    dftotalturma_['evadidos'] = vnumevadidos
    dftotalturma_ = dftotalturma_[dftotalturma_[self.nomevarcurso] != '99999']
    dadoscursos = dftotalturma_
    montargraficoevasao(dftotalturma_,self.nomevarcurso)
    dadoscursos['perc_evadidos'] = dadoscursos.apply(self.__calcula_perc, axis=1)
    display(HTML(dadoscursos.head().to_html()))

  def plota_estat_dados(self,df_):
    self.__verificadados(df_,config_experimento.featuresselecionadas)
    display(HTML(df_.loc[:,config_experimento.featuresselecionadas].describe().to_html()))
    plota_heatmap(df_.loc[:,config_experimento.featuresselecionadas])

  def plota_hist_dados(self,df_):
    histplotX(df_.loc[:,config_experimento.featuresselecionadas])

#######################################################################################################################
# Classes composite para Pré-processamento
#######################################################################################################################
import abc
class Component(metaclass=abc.ABCMeta):

    def __init__(self):
      self.silent = 'S'
    
    def get_silent(self):
      return self.silent
    
    def set_silent(self,valor):
      self.silent = valor

    @abc.abstractmethod
    def realiza_operacao(self,X_,y_):
        pass

class Preprocessamento(Component):

    def __init__(self):
        super().__init__()
        self._children = []
    
    def realiza_operacao(self,X_,y_):
        for child in self._children:
          child.silent = super().get_silent()
          X_,y_ = child.realiza_operacao(X_,y_)

        return X_,y_

    def add(self, component):
      idx = [i for i, j in
           enumerate(self._children) if type(component) is type(j)]
      if len(idx) > 0:
        self._children[idx[0]]=component
      else:
        self._children.append(component)

    def remove(self, component):
      idx = [i for i, j in
           enumerate(self._children) if type(component) is type(j)]
      if len(idx) > 0:
        self._children.pop(idx[0])
    
    def retira_acento(self,frase_):
      nome = frase_.upper()
      retiraacentos = [
          ['.', ''],['Á','A'],['À','A'],['Ê','E'],['Ô','O'],['Û','U'],
          ['Õ','O'],['É','E'],['Í','I'],['Ç','C'],['Ü','U'],['Ä','A'],
          ['Â','A'],['Ã','A'],['Ó','O'],['Ú','U']
      ]

      for acento in retiraacentos:
        nome = nome.replace(acento[0],acento[1])

      return nome
   
    def apaga_instancias(self,nomecampo,conteudo,df_):
      df_ = df_.drop(df_[df_[nomecampo].str.strip() == conteudo].index, axis=0)
      return df_

    def renomear_conteudo_campo(self,nome_ant,nome_novo,df_,nomecampo='curso'):
      df_[nomecampo] = df_[nomecampo].replace(nome_ant,nome_novo)      
      return df_

#######################################################################################################################
# Classe para detecção de outliers
#######################################################################################################################
"""(Detectando Outlier com Boxplot e biblioteca PyOD)"""
class Engaja_outliers(Component):

  def __init__(self,df = None):
    super().__init__()
    self.lista_outliers = None
    self.__outliers = []
    self.__df_ = df
    self.__X_ = self.__df_.loc[:,config_experimento.featuresselecionadas]
    self.__y_ = self.__df_.loc[:,config_experimento.var_y]
    # if (self.__df_ is None):
    #   raise Exception("DataFrame Não pode estar vazio")
  
  def __definepca(self,ncomp,X_):
    pca = PCA(n_components=ncomp)
    x = StandardScaler().fit_transform(X_)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal_component_1', 'principal_component_2'])
    print(pca.explained_variance_ratio_)
    return principalDf
  
  def __define_outlier(self):
    # Distância Euclidiana
    knn_pipeline = Pipeline(steps=[
      ("normalizacao", MinMaxScaler()),
      ("KNN", KNN())
    ])
    detector = knn_pipeline
    detector.fit(self.__X_)
    return detector
  
  def __verifica_outlier_dados(self):
    detector = self.__define_outlier()
    ## Plota dispersão da base original
    finalDf = pd.concat([self.__definepca(2,self.__X_), self.__y_], axis = 1)
    print("Dispersão da base original")
    plotadispersao(finalDf)
    previsoes_outlier = detector['KNN'].labels_
    for i in range(len(previsoes_outlier)):
      if previsoes_outlier[i] == 1:
        self.__outliers.append(i)
    self.lista_outliers = self.__df_.iloc[self.__outliers,:]
    #Remoção de outlier e nova plotagem da base original
    Xtemp = self.__df_.drop(self.__outliers,axis=0).loc[:,config_experimento.featuresselecionadas]
    ytemp = self.__df_.drop(self.__outliers,axis=0)[config_experimento.var_y]
    
    finalDf = pd.concat([self.__definepca(2,Xtemp), ytemp], axis = 1)
    print("Dispersão depois de remoção de outliers")
    plotadispersao(finalDf)
    #return self.lista_outliers,self.__outliers
  
  def salvar_lista_outliers(self,nomearquivo='outliers.xlsx'):
    # Salva Lista de Outlier
    if self.lista_outliers is None:
      print('Sem dados para salvar. Chame o método "realiza_operacao" '+
      'para gerar a lista')
    else:
      self.lista_outliers.to_excel(nomearquivo,float_format="%.3f", encoding = 'utf8')
      print('Lista de outliers salva com sucesso no arquivo "'+nomearquivo+'"')
  
  def remove_outliers(self):
    if config_experimento.remove_outlier:
      return self.__df_.drop(self.__outliers,axis=0)
    else:
      return self.__df_
  
  def realiza_operacao(self,X_,y_):
    if config_experimento.remove_outlier:
      print("\n-----Detecção e remoção de outliers-----")
      self.__X_ = X_
      self.__y_ = y_
      self.__verifica_outlier_dados()
      print("Concluído!")
      return self.__X_.drop(self.__outliers,axis=0),self.__y_.drop(self.__outliers,axis=0)
    else:
      return X_,y_

#######################################################################################################################
# Classes de balanceamento e normalização de dados
#######################################################################################################################
class Engaja_balanceamento(Component):

  def __init__(self):
    super().__init__()
    
  # Oversampling
  def realiza_operacao(self,X_,y_):
    if config_experimento.combalanceamento:
      if super().get_silent() == 'N':
        print("\n-----Balanceamento dos dados com SMOTE-----")
      smote = SMOTE(sampling_strategy='auto',random_state=2)
      X_over, y_over = smote.fit_resample(X_,y_)
      if super().get_silent() == 'N':
        print(Counter(y_),Counter(y_over))
        print("Concluído!")
      return X_over, y_over
    else:
      if super().get_silent() == 'N':
        print("\n-----Sem balanceamento de dados configurado -----")
      return self.X_, self.y_

class Engaja_normalizacao(Component):

  def __init__(self,tp=1):
    super().__init__()
    self.tp = tp

  def realiza_operacao(self,X_,y_):
    if config_experimento.comnormalizacao:
      print("\n-----Normalização dos dados para treinamento-----")
      columns_to_standardize = config_experimento.featuresselecionadas
      if self.tp == 1:
        min_max_scaler = MinMaxScaler()
      else:
        min_max_scaler = RobustScaler()

      for column in columns_to_standardize:
        X_[column] = min_max_scaler.fit_transform(
            np.array(X_[column]).reshape(-1,1))
      print("Concluído!")
    else:
      print("\n-----Sem normalização configurada para este classificador-----")

    return X_,y_

#######################################################################################################################
# Classe XAI com Shap (Gráficos)
#######################################################################################################################
class Engaja_Xai:

  def __init__(self,modelo,X_train,y_train):
    self.normalizamodelo = False
    self.__modelo  = modelo
    self.__normalizacao = Engaja_normalizacao()
    self.__X_train = X_train
    self.__y_train = y_train
    self.__flgexplainer = 0    
    plt.rcParams.update({'font.family' :'sans-serif',
                                'font.weight': 'bold','font.size': 22})
  
  def __salvarfigura(self,nome,formato,plt,dpi=None):
    plt.savefig(nome+"."+formato,bbox_inches="tight",dpi=dpi,format=formato)
  
  ################ Aplica SHAP no modelo treinado ##################
  def efetua_xaimodelo_global(self,section,nomemodelo,complots='S'):
    print("\n-----Efetuando cálculo de Shap Values para os "+
          "dados de treinamento-----")
    
    self.__vparam_shap = config_experimento.retval_cfg_modelo(nomemodelo,
                                                            'params_shap')
    print("Modelo: ",nomemodelo)
    check_additivity=self.__vparam_shap['check_additivity']

    if self.__vparam_shap['tipo_explainer'] != 'tree':
      self.__explainer = shap.Explainer(self.__modelo.predict_proba, self.__X_train,
                                feature_names=self.__X_train.columns)
      self.__shap_values = self.__explainer(self.__X_train)
      self.__shap_valuesw = self.__shap_values
    else:
      self.__explainer = shap.TreeExplainer(self.__modelo,
                                          self.__X_train,
                                          model_output=
                                          self.__vparam_shap['param_proba'])
      self.__shap_values = self.__explainer(self.__X_train,
                                        check_additivity=check_additivity)
      self.__shap_valuesw = self.__explainer.shap_values(self.__X_train)
    if complots == 'S':
      self.plota_sumario_beeswarm(section,tpgraph=0)
      self.plota_sumario_bardend(section,tpgraph=0)
    self.__flgexplainer = 1
    print("Concluído!")

################ Análise importância Local com Plots ##################
  def __plota_xaimodelo_local(self,instancia,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,'Análise Importância Local',
                                   complots='N')
    shap_values = self.__explainer(instancia)

    fig = plt.figure(figsize=(200, 150))
    if self.__vparam_shap['dim_shapvalues'] == 1:
      shap.plots.force(shap_values[0,:,1],
                       matplotlib=True,show=False,text_rotation=12)
      if tpgraph == 1:
        self.__salvarfigura("xaiind",frtsaving,plt,
                   600 if frtsaving == 'png' else None)
      plt.show()
      shap.plots.waterfall(shap_values[0,:,1],
                           show=False,max_display=15)
      if tpgraph == 1:
        self.__salvarfigura("xaiind_waterfall",frtsaving,plt,
                   600 if frtsaving == 'png' else None)
      plt.show()
    else:
      shap.plots.force(shap_values[0],
                       matplotlib=True,show=False,text_rotation=12)
      self.__salvarfigura("xaiind",frtsaving,plt,
                   600 if frtsaving == 'png' else None)
      plt.show()
      shap.plots.waterfall(shap_values[0],
                           show=False,max_display=15)
      self.__salvarfigura("xaiind_waterfall",frtsaving,plt,
                   600 if frtsaving == 'png' else None)
      plt.show()

  ########################
  def executa_xai_local(self,nomecampo,conteudocampo,dfpredicteste_,
                        frtsaving,tpgraph=1,campodisciplina='disciplina'):
    vdadosnovosaluno = dfpredicteste_[dfpredicteste_[nomecampo] == 
                                      conteudocampo]

    if self.normalizamodelo:
      self.__normalizacao.silent ='S'
      config_experimento.comnormalizacao = self.normalizamodelo
      vdadosnovosaluno = self.__normalizacao(vdadosnovosaluno,None)

    for (i,row) in vdadosnovosaluno.iterrows():
      print('Pesquisa de',nomecampo,':',
            conteudocampo,'Disciplina: ',row[campodisciplina])
      self.__plota_xaimodelo_local(
          row[config_experimento.featuresselecionadas].to_frame().T,frtsaving)
      
  ###################### Plots Análise importância Global #####################
  def __make_shap_waterfall_plot(self,shap_values, features, num_display=20):
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / 
                     np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]

    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4

    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))

  ##################
  def plota_sumario_beeswarm(self,section,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,section,complots='N')

    if self.__vparam_shap['dim_shapvalues'] == 1:
      shap.summary_plot(self.__shap_values[:,:,1], 
                        self.__X_train,plot_type='violin',show=False)
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_beeswarm_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()
    else:
      shap.summary_plot(self.__shap_values,self.__X_train,plot_type='violin',
                        show=False,plot_size=[10.5,7])
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_beeswarm_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()

  ##################
  def plota_sumario_bar(self,section,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,section,complots='N')

    if self.__vparam_shap['dim_shapvalues'] == 1:
      shap.summary_plot(self.__shap_values[:,:,1], self.__X_train,
                        plot_type='bar',show=False)
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_summaryplot_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()
    else:
      shap.summary_plot(self.__shap_values, self.__X_train,plot_type='bar',
                        show=False,plot_size=[10.5,7])
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_summaryplot_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()

  ##################
  def plota_sumario_bardend(self,section,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,section,complots='N')
    
    clust = shap.utils.hclust(self.__X_train,self.__y_train);
    if self.__vparam_shap['dim_shapvalues'] == 1:
      shap.plots.bar(self.__shap_values[:,:,1], clustering=clust,show=False);
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_dendograma_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()
    else:
      shap.plots.bar(self.__shap_values, clustering=clust,show=False);
      if tpgraph == 1:
        self.__salvarfigura("Xaiglogal_dendograma_"+section,frtsaving,plt,
                    600 if frtsaving == 'png' else None)
      plt.show()

  ##################
  def plota_waterfall_impacto(self,section,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,section,complots='N')
    
    if self.__vparam_shap['dim_shapvalues'] == 0:
      self.__make_shap_waterfall_plot(self.__shap_valuesw, self.__X_train)
    else:
      self.__make_shap_waterfall_plot(self.__shap_valuesw[1], self.__X_train)
    if tpgraph == 1:
      self.__salvarfigura("Xaiglogal_waterfallplot_"+section,frtsaving,plt,
                  600 if frtsaving == 'png' else None)
    plt.show()
    
  ##################
  def plota_scatter_corrtarget_var(self,section,tpgraph=1,frtsaving = 'png'):
    if self.__flgexplainer == 0:
      self.efetua_xaimodelo_global(self,section,complots='N')

    vnrows = math.ceil(len(self.__X_train.columns)/4)
    fig, axes = plt.subplots(nrows = vnrows, ncols = 4)
    axes = axes.flatten()
    fig.set_size_inches(40, 20)
    for ax, col in zip(axes, self.__X_train.columns):
      if self.__vparam_shap['dim_shapvalues'] == 1:
        shap.plots.scatter(self.__shap_values[:,col,1],show=False,ax = ax)
      else:
        shap.plots.scatter(self.__shap_values[:,col],show=False,ax = ax)
    if tpgraph == 1:
      self.__salvarfigura("Xaiglogal_scatterplot_"+section,frtsaving,plt,
                  600 if frtsaving == 'png' else None)
    plt.show()

#######################################################################################################################
#######################################################################################################################
# Classe principal para o Arcabouço Engajamais
#######################################################################################################################
#######################################################################################################################
class Engaja_mais:

  def __init__(self):
    self.analise_expl_dados = Engaja_analise_Expl()
    self.preprocessamento = Preprocessamento()
    self.df_results_crossval = definedf_result()
    self.resultadosvalg=[]
    self.resultbestmodel=[]
    self.modelosvalidados = []
    self.modeloxai = None
    self.analise_Xai = None
    self.df_ = None
    self.X_ = None
    self.y_ = None
    self.X_train = None
    self.dfpredicteste = None
    self.reset_crossval = True
    self.reg_hiperparams = pd.DataFrame()
    self.__balanceamento = Engaja_balanceamento()
    self.__normalizacao = Engaja_normalizacao()
    self.__flgtreinorealiz = 0
    self.__vnormmodeltr = None
    self.__modelotreinado = None

  ##### Funções para manipulação de arquivo #####
  def carrega_arq_treino(self,nomearquivo,tipoarquivo='csv',separador=';',
                      encod='utf8'):
    dftemp = None
    if tipoarquivo == 'csv':
      dftemp = pd.read_csv(nomearquivo,sep=separador,encoding=encod)
    elif tipoarquivo == 'xlsx':
      dftemp = pd.read_excel(nomearquivo)
    if dftemp is None:
      raise Exception("Tipo de arquivo desconhecido. Tente csv ou xlsx")
    else:
      #print(dftemp.head().to_markdown())
      display(HTML(dftemp.head().to_html()))
      self.df_ = dftemp
      # Composite para preprocessamento
      self.__remove_outliers = Engaja_outliers(self.df_)
      self.preprocessamento.add(self.__remove_outliers)
      self.preprocessamento.add(self.__balanceamento)
      self.preprocessamento.add(self.__normalizacao)
      self.separaXy()
      self.resultadosvalg=[]
      self.resultbestmodel=[]
      self.reset_crossval = True
      print(self.X_.columns)
  
  def separaXy(self):
      self.X_ = self.df_.loc[:,config_experimento.featuresselecionadas]
      self.y_ = self.df_[config_experimento.var_y]

  ########################
  def salva_df(self,dataframe,nomearq,tipoarquivo='xlsx'):
    if tipoarquivo == 'xlsx':
      dataframe.to_excel(nomearq,float_format="%.3f", index=False)
    else:
      dataframe.to_csv(nomearq,index=False)

  def gravametricasf1_score(self,pastadest='', nomearq='UNIVALE'):
      #### Salva os melhores hiperparâmetros da validação cruzada para todos os modelos tesados ###
      self.salva_df(self.reg_hiperparams,
                             pastadest + 'df_best_hiperpar_resultados_' + nomearq + '.xlsx')
      #### Salva os resultados da validação cruzada para todas as iterações de todos os modelos testados ###
      vtodostestes = pd.DataFrame({self.modelosvalidados[i]: self.resultadosvalg[i]
      ['f1_test'] for i in range(0, len(self.modelosvalidados))})
      self.salva_df(vtodostestes,
                             pastadest + 'df_todos_resultados_' + nomearq + '.xlsx')
      #### Salva os melhores resultados da validação cruzada para todos os modelos testados ###
      vbesttestes = pd.DataFrame({self.modelosvalidados[i]: self.resultbestmodel[i]
      ['f1_score'] for i in range(0, len(self.modelosvalidados))})
      self.salva_df(vbesttestes,
                             pastadest + 'df_best_resultados_' + nomearq + '.xlsx')

  def exibe_estat_metricas(self):
    #### Verifica se tem algum modelo já validado
    if self.modelosvalidados != []:
        resultmetricas = self.resultbestmodel
        df_metricas = pd.DataFrame({self.modelosvalidados[i]: resultmetricas[i]
        ['f1_score'] for i in range(0, len(self.modelosvalidados))})
        display(HTML(df_metricas.describe().T.to_html()))


  ########################
  def salva_modelo(self,filename):
    if not(self.modeloxai is None):
      pickle.dump(self.modeloxai, open(filename, 'wb'))
      self.X_train.to_csv('X_train.csv',index=False)
    else:
      print("O modelo não foi gerado!")

  ########## Teste de Wilcoxon com a métrica passada no parâmetro ###########
  def exec_testwilcoxon(self,metrica):
    for i in range(0,len(self.modelosvalidados)):
      for ix in range(0,len(self.modelosvalidados)):
        if (self.modelosvalidados[i] != 
            self.modelosvalidados[ix]):
          print(self.modelosvalidados[i],
                "->",self.modelosvalidados[ix],
                wilcoxon(self.resultadosvalg[i][metrica],
                        self.resultadosvalg[ix][metrica]))

  ########## Treino o modelo com todos os dados disponíveis ###########
  def exec_treino_modelo(self,nomemodelo=None,section="Seção de Treino",params=None):
      modeloxai = None
      modelogrid = None
      self.separaXy()

      for vclassificador in config_experimento.modelos:
        nome = vclassificador['nome_classificador']
        self.modeloxai = copy.deepcopy(vclassificador['classificador'])
        param = vclassificador['parametros']
        vnorm = vclassificador['normaliza']
        if nome == nomemodelo:
          print("Dados Normalizados: ",vnorm)
          config_experimento.comnormalizacao = vnorm
          
          self.preprocessamento.set_silent('N')
          X_train,y_train = self.preprocessamento.realiza_operacao(
              self.X_,self.y_)
          
          if params != None:
            self.modeloxai.set_params(**params)
            print(params)
          else:
            if config_experimento.otimizahiperparametros:
              modelogrid = executargridsearch(self.modeloxai,param,
                                              X_train,y_train)
              self.modeloxai = modelogrid.best_estimator_
      
      print(self.modeloxai)
      self.modeloxai.fit(X_train,y_train)
      self.analise_Xai = Engaja_Xai(self.modeloxai,X_train,y_train)
      self.analise_Xai.efetua_xaimodelo_global(section,nomemodelo,'N')
      self.analise_Xai.normalizamodelo = config_experimento.comnormalizacao
      self.X_train = X_train
      self.__vnormmodeltr = vnorm
      self.__modelotreinado = nomemodelo

  ########################
  def exec_predicao_modelo(self,dfteste_):
    if not (self.modeloxai is None):
      print(self.modeloxai)
      X_testpred = dfteste_.loc[:,config_experimento.featuresselecionadas]
      y_testpred = dfteste_[config_experimento.var_y]
      plt.rcParams.update(plt.rcParamsDefault)
      config_experimento.comnormalizacao = self.__vnormmodeltr

      X_testpred,y_testpred = self.__normalizacao.realiza_operacao(
          X_testpred.copy(),y_testpred.copy())  
      
      previsao_mod = self.modeloxai.predict(X_testpred)
      y_pred_prob = self.modeloxai.predict_proba(X_testpred)
      matriz_conf = matriz_confusao(y_testpred, previsao_mod,
                                    self.modeloxai,self.__modelotreinado)
      temp_name = self.__modelotreinado+"_conf_matrix_pred_dadosnovos.png"
      matriz_conf.savefig(temp_name)
      print(classification_report(y_testpred,previsao_mod))
      vtestes = classification_report(y_testpred,previsao_mod,output_dict=True)
      # resultado_pred_real_model.append(vtestes)
      vauc = roc_auc_score(y_testpred, previsao_mod)
      print("ROC-AUC ",vauc)

      # Juntando as predições com o dataset original
      dadospred = {'prediction': previsao_mod,
        'proba_1': y_pred_prob[:,1]}

      predictions = pd.DataFrame(dadospred)
      
      self.dfpredicteste = pd.concat(
          [predictions.reset_index(drop=True),
           dfteste_.reset_index(drop=True)], axis=1)
      self.dfpredicteste["proba_1"] = self.dfpredicteste[
          "proba_1"].map('{:.4f}'.format)
      print("Concluído!")
    else:
      print('Nenhum modelo treinado! Execute a Função "treina_modelo"')

  ########## Executa CrossValidation nos dados de treinamento ###########
  def exec_crossval_dados(self,section="Experimento I",complotiter = 'S'):
    if self.reset_crossval:
      self.df_results_crossval = definedf_result()
      self.resultadosvalg=[]
      self.resultbestmodel=[]
      self.reg_hiperparams = pd.DataFrame()
      self.modelosvalidados = []
    
    self.separaXy()
    vnomemodelo = []
    vhiperparametro = []
    nomes = []
    X_train,y_train = self.__remove_outliers.realiza_operacao(self.X_.copy(),
                                                            self.y_.copy())

    for vclassificador in config_experimento.modelos:
      nome = vclassificador['nome_classificador']

      if vclassificador['detalhe_treino']:
        self.__balanceamento.silent = 'N'
      else:
        self.__balanceamento.silent = 'S'

      if not(self.__verificacrossval_modelo(nome)):
        modelo = copy.deepcopy(vclassificador['classificador'])
        params = vclassificador['parametros']
        vnorm = vclassificador['normaliza']
        if vclassificador['executa_validacao']:
          self.modelosvalidados.append(vclassificador['nome_classificador'])
          print(nome,"-Dados Normalizados: ",vnorm)
          varprocess  = {'vf1modelkfold':[],'vprecmodelkfold':[],
                        'vrecallmodelkfold':[],'vaccmodelkfold':[],
                        'vrocaucmodelkfold':[],'vf1modelkfoldg':[],
                        'vacuracia':[],'vprecisao':[],'vrecall':[],'vf1':[],
                        'vroc':[],'vrecallmacro':[]}
          config_experimento.comnormalizacao = vnorm
          X_train,y_train = self.__normalizacao.realiza_operacao(
              X_train.copy(),y_train.copy())  

          for i in range(30):
            print('Iteração:',i+1,"/",30)
            X_cross,X_test_,y_cross,y_test_ = train_test_split(X_train,y_train,
                    test_size=config_experimento.perc_test,
                    random_state=i,shuffle=True,stratify=y_train)
          
            X_cross,y_cross = self.__balanceamento.realiza_operacao(
                X_cross.copy(),y_cross.copy())

            vhiper_temp = None
            if config_experimento.otimizahiperparametros:
                modelgs = executargridsearch(modelo,
                                          params,X_cross,
                                          y_cross,config_experimento.scores,'S')
                modelo = modelgs.best_estimator_
                vhiper_temp = {'modelo':nome,
                                      'hiperparam':str(modelgs.best_params_)}

            else:
                vhiper_temp = {'modelo':nome,
                                      'hiperparam':str(modelo.get_params())}

            vnomemodelo.append(nome)

            cv_results = evaluate_model(modelo, X_cross,
                                          y_cross,config_experimento.scores,
                                          n_splits=config_experimento.num_folds,
                                          n_repeats=config_experimento.n_repeats)

            for i in cv_results['test_accuracy']:
              varprocess['vacuracia'].append(i)
            for i in cv_results['test_precision']:
              varprocess['vprecisao'].append(i)
            for i in cv_results['test_recall']:
              varprocess['vrecall'].append(i)
            for i in cv_results['test_f1']:
              varprocess['vf1'].append(i)
            for i in cv_results['test_roc_auc']:
              varprocess['vroc'].append(i)
            for i in cv_results['test_recall_macro']:
              varprocess['vrecallmacro'].append(i)

            vmelhormetricf1 = []
            vmelhormetricprec = []
            vmelhormetricrecall = []
            vmelhormetricacc = []
            vmelhormetricrocauc = []
            for vestm in cv_results['estimator']:
              y_predestm = vestm.predict(X_test_)
              vresmetric=classification_report(y_test_,y_predestm,
                                                output_dict=True)
              varprocess['vf1modelkfoldg'].append(
                  vresmetric['macro avg']['f1-score'])

              vmelhormetricf1.append(vresmetric['macro avg']['f1-score'])
              vmelhormetricprec.append(vresmetric['macro avg']['precision'])
              vmelhormetricrecall.append(vresmetric['macro avg']['recall'])
              vmelhormetricacc.append(vresmetric['accuracy'])
              vmelhormetricrocauc.append(roc_auc_score(y_test_,y_predestm))


            varprocess['vf1modelkfold'].append(max(vmelhormetricf1))
            varprocess['vprecmodelkfold'].append(max(vmelhormetricprec))
            varprocess['vrecallmodelkfold'].append(max(vmelhormetricrecall))
            varprocess['vaccmodelkfold'].append(max(vmelhormetricacc))
            varprocess['vrocaucmodelkfold'].append(max(vmelhormetricrocauc))
            if vclassificador['detalhe_treino']:
              print("Média Crossval f1-score: {:.5f}".
                  format(np.mean(cv_results['test_f1'])),
                  "   Melhor Crossval f1-score......: {:.5f}".
                  format(max(cv_results['test_f1'])))
              print("Média teste f1-score...: {:.5f}".
                  format(np.mean(vmelhormetricf1)),
                  "   Melhor predição teste f1-score: {:.5f}".
                  format(max(vmelhormetricf1)))

            vhiper_temp['best_train'] = max(cv_results['test_f1'])
            vhiper_temp['best_test'] = max(vmelhormetricf1)
            vhiperparametro.append(vhiper_temp)

          print('{:<24s} {:>17s} {:>17s} {:>17s} {:>17s} {:>10s} {:>17s}'.
            format("Modelo com Média/Mediana","Acurácia","Precisão","Recall","f1","Std_f1","ROC/AUC"))
          nomes.append(nome)
          msg = "{:<24s} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f} {:10.4f}/{:6.4f}".format(nome,
                                  np.mean(varprocess['vacuracia']),
                                  np.median(varprocess['vacuracia']),
                                  np.mean(varprocess['vprecisao']),
                                  np.median(varprocess['vprecisao']),
                                  np.mean(varprocess['vrecall']),
                                  np.median(varprocess['vrecall']),
                                  np.mean(varprocess['vf1']),
                                  np.median(varprocess['vf1']),
                                  np.std(varprocess['vf1']),
                                  np.mean(varprocess['vroc']),
                                  np.median(varprocess['vroc']))
          print(msg)
          msg = "{:<24s} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f}/{:6.4f} {:10.4f} {:10.4f}/{:6.4f}".format(
                                  nome+" Teste",
                                  np.mean(varprocess['vaccmodelkfold']),
                                  np.median(varprocess['vaccmodelkfold']),
                                  np.mean(varprocess['vprecmodelkfold']),
                                  np.median(varprocess['vprecmodelkfold']),
                                  np.mean(varprocess['vrecallmodelkfold']),
                                  np.median(varprocess['vrecallmodelkfold']),
                                  np.mean(varprocess['vf1modelkfold']),
                                  np.median(varprocess['vf1modelkfold']),
                                  np.std(varprocess['vf1modelkfold']),
                                  np.mean(varprocess['vrocaucmodelkfold']),
                                  np.median(varprocess['vrocaucmodelkfold']))
          
          set_scores_result(self.df_results_crossval, varprocess, nome)

          print(msg)

          self.resultadosvalg.append({'modelo':nome,'f1': varprocess['vf1'],
                                      'f1_test': varprocess['vf1modelkfoldg']})
          self.resultbestmodel.append({'modelo':nome,'f1_score': varprocess['vf1modelkfold'],
                                       'accuracy': varprocess['vaccmodelkfold'],
                                       'precision':varprocess['vprecmodelkfold'],
                                       'recall': varprocess['vrecallmodelkfold'],
                                       'roc_auc': varprocess['vrocaucmodelkfold']})

          if complotiter == 'S':
            plotresult(varprocess['vf1'],varprocess['vf1modelkfoldg'],"f1-score",section)
       
    if not(vnomemodelo == []):
      self.reg_hiperparams = self.reg_hiperparams.append(vhiperparametro,
                                                             ignore_index=True, 
                                                             sort=False)
    self.__flgtreinorealiz = 1

  ########## Retorna o melhor conjunto de hiperparâmetros e seu respectivo modelo ###########
  def retorna_bestmodel(self):
    vnomemodel=""
    if self.__flgtreinorealiz == 0:
        print("Nenhum Modelo Validado")
        return []
    else:
        vbestmodel = self.df_results_crossval.reset_index(level=['CLF', 'SCORE'])
        vnomemodel = vbestmodel[vbestmodel['SCORE'] == 'f1'].groupby(['CLF'],
                                 as_index=False)[1].sum().sort_values(1, ascending=False).head(1)['CLF'].values[0]
        self.reg_hiperparams['qtde'] = 0
        vbestparams = self.reg_hiperparams[self.reg_hiperparams['modelo'] == vnomemodel].groupby(
            ['modelo','hiperparam'],
            as_index=False)['best_test'].mean().sort_values(
            ['modelo','best_test'],
            ascending=False).head(1)['hiperparam'].values[0]

    return [vnomemodel,vbestparams]

  ###################
  def __verificacrossval_modelo(self,nomemodelo):
    retorno = False
    for modelo in self.resultadosvalg:
      if modelo['modelo'] == nomemodelo:
        retorno = True
    return retorno

  ###################
  def plota_iteracoescrossval(self,section,nomemodelo):
    for modelo in self.resultadosvalg:
      if modelo['modelo'] == nomemodelo:
        plotresult(modelo['f1'],modelo['f1_test'],"f1-score",section)
  
  ###################
  '''tp = 1, plota a mediana de todos os testes. tp = 2 plota a mediana dos melhores hiperparametros'''
  def plota_boxplotcrossval(self,section,tp=2,df_param=None):

    if df_param is None:
        if self.__flgtreinorealiz == 0:
            self.executa_cross_val_dados(section)

        if tp == 1:
            boxplot_sorted(self.modelosvalidados,self.resultadosvalg,
                       'f1',figsize=(25,10),
                      fontsize=24,rot=90,section=section,by=['Modelo'])
            boxplot_sorted(self.modelosvalidados,self.resultadosvalg,
                       'f1_test',figsize=(25,10),
                      fontsize=24,rot=90,section=section,by=['Modelo'])
        else:
            boxplot_sorted(self.modelosvalidados, self.resultbestmodel,
                           'f1_score', figsize=(25, 10),
                           fontsize=24, rot=90, section=section, by=['Modelo'])
    else:
        boxplot_sorted(self.modelosvalidados, self.resultbestmodel,
                       'f1_score', figsize=(25, 10),
                       fontsize=24, rot=90, section=section, by=['Modelo'], df_definido=df_param)



  #####Executa método Wrapper BorutaShap para importância de atributos#####
  def exec_wrapper_borutashap(self):
      config_experimento.comnormalizacao = self.__vnormmodeltr
      self.separaXy()

      X_train,y_train = self.__normalizacao.realiza_operacao(
          self.X_.copy(),self.y_.copy())
      selector = BorutaShap(model=self.modeloxai,
                            importance_measure = 'shap', classification = True)
      
      print('Realizando cálculo de importância de features - BorutaShap')
      selector.fit(X = X_train, y = y_train,
                   n_trials = 200, sample = False, verbose = True)
      
      print('Features para remoção')
      selector.features_to_remove

      print('Boxplot de importância de Features')
      selector.plot(which_features='all', figsize=(16,12))

  ########### Apresenta os melhores hiperparâmetros do treinamento ##########  
  def lista_melhores_hiperp(self):
    # Melhores hiperparametros
    self.reg_hiperparams['qtde'] = 0
    display(HTML(self.reg_hiperparams.groupby(['modelo','hiperparam'],
        as_index=False)['best_test'].mean().sort_values(
            ['modelo','best_test'], ascending=False).to_html()))

#######################################################################################################################
# Definição das Configurações padrões
#######################################################################################################################
config_experimento.featuresselecionadas = ['med_ac_lti_sema_disc','tmp_medutil_semanahr','med_geral_ac_sema_aluno',
                        'med_ac_lti_aluno_sema_disc',
                        'qtde_reprov_disc',
                        'qtde_rep_prim_modulo','qtde_aprov_prim_modulo','qtde_rep_curso',
                        'qtde_reprov_ult_modulo','qtde_trancamento_curso','qtde_evasao_instituicao',
                        'qtde_cursos_concluidos_instituicao','idade',
                        'med_difip_ac_sema_aluno','sexo','periodo_modulo','med_ac_manha_sema_aluno','med_ac_tarde_sema_aluno',
                        'med_ac_noite_sema_aluno','med_ac_madruga_sema_aluno']

config_experimento.scores = ['accuracy','recall','f1','precision','roc_auc','recall_macro']

### Define os classificadores e suas configurações para uso no arcabouço ####
reghiperparams = pd.DataFrame()
config_experimento.adiciona_modelo({'nome_classificador':'LR',
                'classificador':LogisticRegression(), 
                'parametros': {
                    'max_iter': [20, 50, 100, 200, 500, 1000, 2000, 5000],
                    'solver': ['newton-cg','lbfgs', 'sag', 'saga'],
                    'penalty': ['l2', 'none'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'class_weight':['balanced','none']},
                'normaliza':True,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'agnostico',
                    'param_proba':'predict_proba',
                    'check_additivity': True,
                    'dim_shapvalues': 1}})


config_experimento.adiciona_modelo({'nome_classificador':'KNN',
                'classificador': KNeighborsClassifier(), 
                'parametros': {
                    'leaf_size': list(range(1,50)),
                    'n_neighbors': list(range(1,30)),
                    'p': [1,2]},
                'normaliza':False,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'agnostico',
                    'param_proba':'predict_proba',
                    'check_additivity': True,
                    'dim_shapvalues': 1}
                })

config_experimento.adiciona_modelo({'nome_classificador':'RF',
                'classificador': RandomForestClassifier(), 
                'parametros': {'bootstrap': [True, False],
                               'max_depth': [10, 30, 50,None],
                               'min_samples_leaf': [1, 2, 3, 4],
                               'min_samples_split': [2,3,4,5,6],
                               'n_estimators': [100,200,300,400]},
                'normaliza':False,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'tree',
                    'param_proba':'probability',
                    'check_additivity': False,
                    'dim_shapvalues': 0}})

config_experimento.adiciona_modelo({'nome_classificador':'XGB',
                'classificador': XGBClassifier(), 
                'parametros': {
                    "max_depth": [10,30,50],
                    "min_child_weight" : [1,3,6],
                    "n_estimators": [100,200,300,400],
                    "learning_rate": [0.01,0.05, 0.1,0.16,0.2, 1.0]},
                'normaliza':False,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'tree',
                    'param_proba':'predict_proba',
                    'check_additivity': True,
                    'dim_shapvalues': 1}})


config_experimento.adiciona_modelo({'nome_classificador':'CAT',
                'classificador': CatBoostClassifier(logging_level='Silent'), 
                'parametros': {
                    'depth': [4, 7, 10],
                    'learning_rate' : [0.03, 0.1, 0.15],
                    'l2_leaf_reg': [1,4,9],
                    'iterations': [300]},
                'normaliza':False,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'tree',
                    'param_proba':'probability',
                    'check_additivity': True,
                    'dim_shapvalues': 0}})

config_experimento.adiciona_modelo({'nome_classificador':'LGB',
                'classificador':ltb.LGBMClassifier(), 
                'parametros': {
                    "max_depth": [25,50, 75],
                    "learning_rate" : [0.01,0.05,0.1],
                    "num_leaves": [300,600,900,1200],
                    "n_estimators": [200]},
                'normaliza':False,
                'executa_validacao':True,
                'detalhe_treino': True,
                'params_shap': {
                    'tipo_explainer': 'tree',
                    'param_proba':'probability',
                    'check_additivity': True,
                    'dim_shapvalues': 0}})

#######################################################################################################################
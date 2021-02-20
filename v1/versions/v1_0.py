import numpy as np
import pandas as pd
import statistics
import math
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score


##Function that uses the linkage matrix to find potential groupings
def tree(cluster_id, dictionary_of_potential_clusters, list_of_potential_clusters):
    if cluster_id in dictionary_of_potential_clusters:
        if len(dictionary_of_potential_clusters[cluster_id])!=1:
            for new_potential_cluster_id in dictionary_of_potential_clusters[cluster_id]:
                tree(new_potential_cluster_id, dictionary_of_potential_clusters, list_of_potential_clusters)
        else:
            list_of_potential_clusters.append(dictionary_of_potential_clusters[cluster_id])
    else:
        list_of_potential_clusters.append(cluster_id)

##Determine Clusters
def clusters(condensed_matrix, new_df):
    #Create a linkage matrix based upon pairwise distances between (time series/representatives of groups of time series) calculated by the similarity measurement used
    mergings=linkage(condensed_matrix, method='complete')
    dictionary_of_potential_clusters={}
    new_cluster_id=len(new_df)
    for pair in mergings:
        if len(dictionary_of_potential_clusters)!=0 and pair[3]!=2:
            list_of_potential_clusters=[]
            tree(pair[0], dictionary_of_potential_clusters, list_of_potential_clusters)
            tree(pair[1], dictionary_of_potential_clusters, list_of_potential_clusters)
            dictionary_of_potential_clusters[new_cluster_id]=list_of_potential_clusters
        else:
            dictionary_of_potential_clusters[new_cluster_id]=[pair[0],pair[1]]
        new_cluster_id+=1
    for i in range(len(new_df)):
        dictionary_of_potential_clusters[i]=[i]  
    dictionary_of_all_clusters={}
    dictionary_of_potential_cluster_groups={}
    cluster_list=[]
    silhouette_scores=[]
    condensed_matrix=pd.to_numeric(pd.Series(condensed_matrix),downcast='float')
    uncondensed_matrix=squareform(condensed_matrix)
    for potential_number_of_clusters in range(2,len(new_df)):
        dictionary_of_clusters={}
        number=potential_number_of_clusters-1
        cluster_list.append(mergings[-number][0])
        cluster_list.append(mergings[-number][1])
        cluster_list.sort()
        if number!=1:
            cluster_list.remove(cluster_list[-1])
        dictionary_of_potential_cluster_groups[potential_number_of_clusters]=cluster_list.copy()
        for cluster_number in range(potential_number_of_clusters):
            dictionary_of_clusters[cluster_number]=dictionary_of_potential_clusters[dictionary_of_potential_cluster_groups[potential_number_of_clusters][cluster_number]]
        labels=pd.Series(dictionary_of_clusters).explode().sort_values().index
        silhouette_scores.append(silhouette_score(uncondensed_matrix,labels=labels,metric='precomputed'))
        dictionary_of_all_clusters[potential_number_of_clusters]=dictionary_of_clusters.copy()
    optimal_number_of_clusters=silhouette_scores.index(max(silhouette_scores))+2
    return pd.Series(dictionary_of_all_clusters[optimal_number_of_clusters]).explode().sort_values().index
    
##Similarity Measurements
def euclidean(df_ts):
    euclidean_distances=[]
    for i in range(len(df_ts)):
        time_series_i=df_ts.iloc[i]
        for j in range(i+1,len(df_ts)):
            time_series_j=df_ts.iloc[j]
            euclidean_distance=(df_ts.iloc[i]-df_ts.iloc[j])**2
            euclidean_distances.append(math.sqrt(sum(list(euclidean_distance))))
    condensed_euclidean_matrix=np.array(euclidean_distances)
    Euclidean_Series=clusters(condensed_euclidean_matrix,df_ts)
    return Euclidean_Series
            
def r2(df_ts):
    r_squared_values=[]
    for i in range(len(df_ts)):
        time_series_i=df_ts.iloc[i]
        for j in range(i+1,len(df_ts)):
            time_series_j=df_ts.iloc[j]
            correlation_matrix=np.corrcoef(time_series_i,time_series_j)
            r_squared_values.append(1-((correlation_matrix[0,1])**2))
    condensed_r2_matrix=np.array(r_squared_values)
    R2_Series=clusters(condensed_r2_matrix,df_ts)
    return R2_Series

def cosine_similarity(df_ts):
    cosine_similarity_values=[]
    for i in range(len(df_ts)):
        time_series_i=df_ts.iloc[i]
        for j in range(i+1,len(df_ts)):
            time_series_j=df_ts.iloc[j]
            cosine_similarity_values.append(1-((time_series_i@time_series_j.T)/(norm(time_series_i)*norm(time_series_j))))
    condensed_cosine_similarity_matrix=np.array(cosine_similarity_values)
    Cosine_Series=clusters(condensed_cosine_similarity_matrix, df_ts)
    return Cosine_Series

def median(df_ts):
    median_distances=[]
    for i in range(len(df_ts)):
        time_series_i=df_ts.iloc[i]
        for j in range(i+1,len(df_ts)):
            time_series_j=df_ts.iloc[j]
            taxicab_distance=abs(df_ts.iloc[i]-df_ts.iloc[j])
            median_distances.append(statistics.median(taxicab_distance))
    condensed_median_matrix=np.array(median_distances)
    Median_Series=clusters(condensed_median_matrix,df_ts)
    return Median_Series

def vertical(df_ts):
    vertical_distances=[]
    for i in range(len(df_ts)):
        time_series_i=sum(list(df_ts.iloc[i]))/(len(list(df_ts.iloc[i])))
        for j in range(i+1,len(df_ts)):
            time_series_j=sum(list(df_ts.iloc[j]))/(len(list(df_ts.iloc[j])))
            vertical_distances.append(abs(time_series_i-time_series_j))
    condensed_vertical_matrix=np.array(vertical_distances)
    Vertical_Series=clusters(condensed_vertical_matrix,df_ts)
    return Vertical_Series
    

def distance(Y):
    X=Y.copy()
    Cluster_DataFrame=pd.DataFrame(data=[euclidean(X),r2(X),cosine_similarity(X),median(X),vertical(X)]).T
    Cluster_DataFrame['index']=X.index
    Cluster_DataFrame=Cluster_DataFrame.set_index('index')
    
    return Cluster_DataFrame
    
    
    
    
    



##for dataframes that contain groups
def combine_cluster_groups(M,L,dic_1,dic_2,k,final_dict): 
    W=pd.DataFrame.from_dict(dic_1,orient='index')
    Z=W.copy()
    cluster_df=distance(Z)



    cluster_df_compact=cluster_df.copy()
    cluster_df_compact['group']=cluster_df_compact[0].astype('str')+ '_' + cluster_df_compact[1].astype('str')+ '_' + cluster_df_compact[2].astype('str')+ '_' + cluster_df_compact[3].astype('str')+ '_' + cluster_df_compact[4].astype('str')
    cluster_df_compact=cluster_df_compact.drop(columns=[0,1,2,3,4])
    cluster_df_compact=pd.Series(cluster_df_compact['group'], index=cluster_df_compact.index)

    new_cluster_combinations=pd.Series(cluster_df_compact.unique()) #stores new cluster id number and the associated cluster combination

    
    dic_s3={}
    for i in range(len(new_cluster_combinations)):
        for j in range(len(cluster_df_compact)):
            if new_cluster_combinations.iloc[i]==cluster_df_compact.iloc[j]:
                dic_s3[cluster_df_compact.index[j]]=new_cluster_combinations.index[i]
    s3_original=pd.Series(dic_s3)
    s3=pd.Series(data=s3_original.index,index=s3_original.values) #stores new_group id (index) for each old_group id (value)
    

    #dict_3 stores the old_groups (values) within each new_group (key)
    dic_3={}
    for i in range(len(s3)):
        if s3.index[i] not in dic_3:
            dic_3[s3.index[i]]=[s3.values[i]]
        else:
            dic_3[s3.index[i]]+=[s3.values[i]]
    #checking for unique old_groups
    potential_outliers={}
    for i in cluster_df:
        #Outlier_checker allows us to see whether a time series is unique according to any similarity measure
        list_outlier=[]
        outlier_checker=cluster_df[i].value_counts()
        for j in range(len(outlier_checker)):
            if outlier_checker.iloc[j]==1:
                for l in range(len(cluster_df)):
                    if cluster_df[i].values[l]==outlier_checker.index[j]:
                        list_outlier.append(cluster_df.index[l]) #old_groups that are unique 
        potential_outliers[i]=list_outlier
    series_potential_outliers=((pd.Series(potential_outliers)).explode()).value_counts()
    outlier_removal=list(series_potential_outliers[series_potential_outliers>=k].index)

    

    if len(outlier_removal)>=1:
        for i in outlier_removal:
            number=len(final_dict)
            final_dict[number]=dic_2[i]
        Z=Z.drop(outlier_removal, axis=0)
        dic_4={}
        for i in dic_2:
            if i not in outlier_removal:
                dic_4[i]=dic_2[i]
        
        return combine_cluster_groups(M,L,dict(Z.T),dic_4,k,final_dict)
    
    #If all the similarity measurements all agree with each other, then the groups each become clusters and the program stops, since there is no need to further cluster them
    #OR if the number of new_groups is equal to the number of old_groups, then each old_group becomes a cluster and the program stops, since no old_groups were able to be clustered
    max_clusters_per_measure=[]
    for i in cluster_df.columns:
        max_clusters_per_measure.append(len(cluster_df[i].unique()))

    
    if len(dic_2)==max(max_clusters_per_measure) or len(dic_2)==len(Z):
        for i in dic_2:
            number=len(final_dict)
            final_dict[number]=dic_2[i]
        return np.array(pd.Series(final_dict).explode().sort_values().index)

    dic_5={}  
    for i in dic_3:
        for j in dic_3[i]:
            if i not in dic_5:
                dic_5[i]=dic_2[j]
            else:
                dic_5[i]+=dic_2[j]
    

    dic_6={}
    for i in dic_5:
        ts_list=[]
        for j in dic_5[i]:
            ts_list.append(np.array(L.loc[j]))
        group_df=pd.DataFrame(ts_list)
        median_list=[]
        for j in Z.columns:
            median_list.append(statistics.median(group_df[j]))
        dic_6[i]=median_list
    return combine_cluster_groups(M,L,dic_6,dic_2,k,final_dict)





##for the original dataframe
def combine_cluster(df_original,X,k,final_dict):
    cluster_df=distance(X)
    #Create a dictionary that stores time series that are unique according to at least 1 similarity measure
    potential_outliers={}
    for i in cluster_df:
        #Outlier_checker allows us to see whether a time series is unique according to any similarity measure
        list_outlier=[]
        outlier_checker=cluster_df[i].value_counts()
        for j in range(len(outlier_checker)):
            if outlier_checker.iloc[j]==1:
                for l in range(len(cluster_df)):
                    if cluster_df[i].values[l]==outlier_checker.index[j]:
                        list_outlier.append(cluster_df.index[l])  

        
        #list_outlier=list(cluster_df[i][outlier_checker[i]==1].index)  #time series that are unique 
        potential_outliers[i]=list_outlier
    series_potential_outliers=((pd.Series(potential_outliers)).explode()).value_counts()
    outlier_removal=list(series_potential_outliers[series_potential_outliers>=k].index)
    #Each of the time series that is unique is added to a dictionary storing the final clusters (each unique time series becomes an individual cluster)
    for i in outlier_removal:
        number=len(final_dict)
        final_dict[number]=[i]
    X=X.drop(outlier_removal, axis=0)
    #If unique time series were detected (outliers), then the dataset is rescanned to ensure there are no other outliers
    ##(this is important as outliers will influence the representative of each group, since the average is taken, thus must be removed)
    if len(outlier_removal.copy())>=1:
        return combine_cluster(df_original,X,k,final_dict)
    #If there are (no outliers/no more outliers), then groups can be allocated
    #make cluster_df more compact 
    cluster_df_compact=cluster_df.copy()
    cluster_df_compact['group']=cluster_df_compact[0].astype('str')+ '_' + cluster_df_compact[1].astype('str')+ '_' + cluster_df_compact[2].astype('str')+ '_' + cluster_df_compact[3].astype('str')+ '_' + cluster_df_compact[4].astype('str')
    cluster_df_compact=cluster_df_compact.drop(columns=[0,1,2,3,4])
    cluster_df_compact=pd.Series(cluster_df_compact['group'], index=cluster_df_compact.index)

    new_cluster_combinations=pd.Series(cluster_df_compact.unique()) #stores new cluster id number and the associated cluster combination

    
    dic_s3={}
    for i in range(len(new_cluster_combinations)):
        for j in range(len(cluster_df_compact)):
            if new_cluster_combinations.iloc[i]==cluster_df_compact.iloc[j]:
                dic_s3[cluster_df_compact.index[j]]=new_cluster_combinations.index[i]
    s3_original=pd.Series(dic_s3)
    s3=pd.Series(data=s3_original.index,index=s3_original.values)
    
    


    #dict_2 stores the time series's within each group
    dict_2={}
    for i in range(len(s3)):
        if s3.index[i] not in dict_2:
            dict_2[s3.index[i]]=[s3.values[i]]
        else:
            dict_2[s3.index[i]]+=[s3.values[i]]

        
    #If all the similarity measurements all agree with each other, then the groups each become clusters and the program stops, since there is no need to further cluster them
    #OR if the number of groups is equal to the number of time series, then each time series becomes a cluster and the program stops, since no time series were able to be clustered
    max_clusters_per_measure=[]
    for i in cluster_df.columns:
        max_clusters_per_measure.append(len(cluster_df[i].unique()))

    if len(dict_2)==max(max_clusters_per_measure) or len(dict_2)==len(X):
        for i in dict_2:
            number=len(final_dict)
            final_dict[number]=dict_2[i]
        return np.array(pd.Series(final_dict).explode().sort_values().index)
    dict_1={}
    for i in dict_2:
        ts_list=[]
        for j in dict_2[i]:
            ts_list.append(np.array(X.loc[j]))
        group_df=pd.DataFrame(ts_list)
        median_list=[]
        for j in X.columns:
            median_list.append(statistics.median(group_df[j]))
        dict_1[i]=median_list
    return combine_cluster_groups(df_original,X,dict_1,dict_2,k,final_dict)
        
    

def Cluster_Method(X):
    df_original=X.copy()
    Y=df_original.copy()
    combine_dict={}
    dic={}
    for m in range(1,6):
        final_dict={}
        combine_dict[m]=combine_cluster(df_original,Y,m,final_dict)
    return combine_dict



        
        

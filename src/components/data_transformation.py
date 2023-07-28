#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Required Libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import *
from dataclasses import dataclass

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# Set the figure size for better visualization
plt.figure(figsize=(15, 8))

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder


# In[2]:


# Initialize Data Transformation Configuration

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    merged_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')


# # Step 2: Data Transformation

# In[3]:


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initate_data_transformation(self,digital_visits_path, BK_LT_0_path,BK_LT_3_path,BK_LT_3_reserv_ch_path,                                    Canc_LT_3_path,hotel_details_path):
    
        try:
            
            # Replace 'YYYY-MM-DD' with your actual date format in the 'format' parameter
            date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

            digital_visits_df = pd.read_csv(digital_visits_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_0_df = pd.read_csv(BK_LT_0_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_df = pd.read_csv(BK_LT_3_path, parse_dates=['stay_date'], date_parser=date_parser)
            BK_LT_3_reserv_ch_df = pd.read_csv(BK_LT_3_reserv_ch_path, parse_dates=['stay_date'], date_parser=date_parser)
            Canc_LT_3_df = pd.read_csv(Canc_LT_3_path, parse_dates=['stay_date'], date_parser=date_parser)
            hotel_details_df = pd.read_csv(hotel_details_path)
            
            # sample datasets
            print("digital_visits:", digital_visits_df.head())
            print("hotel_details:", hotel_details_df.head())
            print("hotel_bookings_at_leadtime_3_by_reservation_channel:", BK_LT_3_reserv_ch_df.head())
            print("recent_cancellations_at_leadtime_3:", Canc_LT_3_df.head())
            print("bookings_leadtime_0_df:", BK_LT_0_df.head())
            print("bookings_leadtime_3_df:", BK_LT_3_df.head())
            
            # size of relevant datasets
            print("digital_visits shape:", digital_visits_df.shape)
            print("hotel_details shape:", hotel_details_df.shape)
            print("hotel_bookings_at_leadtime_3_by_reservation_channel shape:", BK_LT_3_reserv_ch_df.shape)
            print("recent_cancellations_at_leadtime_3 shape:", Canc_LT_3_df.shape)
            print("bookings_leadtime_0_df shape:", BK_LT_0_df.shape)
            print("bookings_leadtime_3_df shape:", BK_LT_3_df.shape)
            
            # Check for duplicate data
            print("Duplicate Data Summary:")
            print("digital_visits :", digital_visits_df.duplicated().sum())
            print("hotel_details :", hotel_details_df.duplicated().sum())
            print("hotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.duplicated().sum())
            print("recent_cancellations_at_leadtime_3 :", Canc_LT_3_df.duplicated().sum())
            print("bookings_leadtime_0_df :", BK_LT_0_df.duplicated().sum())
            print("bookings_leadtime_3_df :", BK_LT_3_df.duplicated().sum())
                    
            # feature set for relevant datasets
            print("digital_visits features:\n", digital_visits_df.columns,"\n")
            print("hotel_details features:\n", hotel_details_df.columns,"\n")
            print("hotel_bookings_at_leadtime_3_by_reservation_channel features:\n", BK_LT_3_reserv_ch_df.columns,"\n")
            print("recent_cancellations_at_leadtime_3 features:\n", Canc_LT_3_df.columns,"\n")
            print("bookings_leadtime_0_df features:\n", BK_LT_0_df.columns,"\n")
            print("bookings_leadtime_3_df features:\n", BK_LT_3_df.columns,"\n")
            
            # Check data info
            print("Data Information:")

            print("\ndigital_visits :", digital_visits_df.info())
            print("\nhotel_details :", hotel_details_df.info())
            print("\nhotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.info())
            print("\nrecent_cancellations_at_leadtime_3 :", Canc_LT_3_df.info())
            print("\nbookings_leadtime_0_df :", BK_LT_0_df.info())
            print("\nbookings_leadtime_3_df :", BK_LT_3_df.info())
            
            # Check for missing data
            print("Missing Data Summary:")
            print("\ndigital_visits :", digital_visits_df.isnull().sum())
            print("\nhotel_details :", hotel_details_df.isnull().sum())
            print("\nhotel_bookings_at_leadtime_3_by_reservation_channel :", BK_LT_3_reserv_ch_df.isnull().sum())
            print("\nrecent_cancellations_at_leadtime_3 :", Canc_LT_3_df.isnull().sum())
            print("\nbookings_leadtime_0_df :", BK_LT_0_df.isnull().sum())
            print("\nbookings_leadtime_3_df :", BK_LT_3_df.isnull().sum())
            
            
            # Handle Missing Data
            # removing saver_rate columns because of constant value "0"
            BK_LT_3_df.drop(labels=['saver_rate'], axis=1, inplace=True)
            hotel_details_df['trading_area'] = hotel_details_df['trading_area'].fillna('Unassigned')
            hotel_details_df['family_rooms'] = hotel_details_df['family_rooms'].fillna(0)
            hotel_details_df['family_rooms'] = hotel_details_df['family_rooms'].replace(-2146826246,0)
            hotel_details_df['air_conditioned_rooms'] = hotel_details_df['air_conditioned_rooms'].fillna('0')
            hotel_details_df['london_region_split'] = hotel_details_df['london_region_split'].fillna('Other')
            BK_LT_3_df['flex_rate'] = BK_LT_3_df['flex_rate'].fillna('0')
            BK_LT_3_df['semi_flex_rate'] = BK_LT_3_df['semi_flex_rate'].fillna('0')
            logging.info('Handled Missing Data')

            # Handle inconsistencies in air_conditioned_rooms
            hotel_details_df['air_conditioned_rooms'] = hotel_details_df['air_conditioned_rooms'].str.lower()

            # List of substrings to replace
            substrings_to_replace_all_rooms = ['all','fully','whole','rooms']
            substrings_to_replace_no_AC = ['no', '0']

            # Replace values in the DataFrame based on the substrings
            for substr in substrings_to_replace_all_rooms:
                hotel_details_df.loc[hotel_details_df['air_conditioned_rooms'].str.contains(substr), 'air_conditioned_rooms'] = 'Yes'

            for substr in substrings_to_replace_no_AC:
                hotel_details_df.loc[hotel_details_df['air_conditioned_rooms'].str.contains(substr), 'air_conditioned_rooms'] = 'No'
            logging.info('Handled Inconsistency Data')
            
            ### Merge relevant datasets ###
            """
            - digital_visits.csv – This file provides details of digital demand for each site 3 days before the stay date.
            - hotel_details.csv – Includes details of hotels
            - hotel_bookings_at_leadtime_3_by_reservation_channel.csv – This file includes aggregate measures for every hotel 
                                                                        and staydate 3 days before guests are expected to arrive. 
                                                                        The aggregate measures are further broken down by reservation 
                                                                        channels. 
            - recent_cancellations_at_leadtime_3.csv – This file provides details cancellations by site for each stay date,
                                                        3 days before guests arrive. 
            - hotel_bookings_at_leadtime_0.csv – This file provides details of total rooms sold outcome by site and stay date.
            - hotel_bookings_at_leadtime_3.csv – This file includes aggregate measures such as rooms sold, off room (Rooms that were
                                                not available to be sold e.g., refurbishments, maintenance etc), average price information
                                                for flex, saver and semi-flex room products.

            """
            # 'hotel_key' is common keys to merge the digital_visits and hotel_details datasets
            data = pd.merge(digital_visits_df, hotel_details_df,on='hotel_key', how='left')

            # 'hotel_key', 'stay_date','lead_time','total_rooms_sold' are common keys to merge the datasets
            data = pd.merge(data, BK_LT_3_reserv_ch_df, on=['hotel_key', 'stay_date','lead_time'], how='left')

            # 'hotel_key', 'stay_date' are common keys to merge the datasets
            data = pd.merge(data, Canc_LT_3_df, on=['hotel_key', 'stay_date'], how='left')

            # 'hotel_key', 'stay_date','lead_time' are common keys to merge the datasets
            data = pd.merge(data, BK_LT_0_df, on=['hotel_key', 'stay_date','lead_time'], how='left')
            data = pd.merge(data, BK_LT_3_df, on=['hotel_key', 'stay_date','lead_time','total_rooms_sold'], how='left')
            logging.info('Merged Data')
            
            # removing irrelevant and duplicate columns
            columns_to_remove = ['hotel_key', 'lead_time', 'bing_ppc_brand', 'google_ppc_brand', 'hotel_name', 'brand']
            data.drop(labels=columns_to_remove, axis=1, inplace=True)
            logging.info('Removed irrelevant and duplicate columns')
            
            os.makedirs(os.path.dirname(self.data_transformation_config.merged_data_path),exist_ok=True)
            data.to_csv(self.data_transformation_config.merged_data_path,index=False)
            
            ### Exploratory Data Analysis (EDA) for all data
            
            # Replace 'YYYY-MM-DD' with your actual date format in the 'format' parameter
            date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

            # load data
            data = pd.read_csv(self.data_transformation_config.merged_data_path, parse_dates=['stay_date'], date_parser=date_parser)
            
            # size of relevant datasets
            print("shape:", data.shape)
            
            # Check for duplicate data
            print("duplicate data :", data.duplicated().sum())
            
            # handle duplicate data
            data = data.drop_duplicates()
            
            # feature set for relevant datasets
            print("data features:\n", data.columns,"\n")
            
            # Check data info
            print("Data Information:")
            print(data.info())
            
            # Check for missing data
            print("Missing Data Summary:")
            print(data.isnull().sum())
            
            # remove data records where total_rooms_sold values are available
            data = data[~data['total_rooms_sold'].isnull()]
            
            # Check for missing data
            print("Missing Data Summary:")
            print("\ndata :", data.isnull().sum())
            
            # Handle missing data
            columns_to_fill = ['Canxrooms_last7days', 'off_rooms', 'rooms_remaining',                       'flex_rate','semi_flex_rate','sellable_capacity','finalroomssold']
            fill_value = 0  # Replace missing values with 0
            data[columns_to_fill] = data[columns_to_fill].fillna(0)

            # Check for missing data
            print("Missing Data Summary:")
            print("\ndata :", data.isnull().sum())
            
            ### Feature Engineering ###
            
            # removing irrelevant and duplicate columns
            columns_to_remove = ['finalroomssold']
            data.drop(labels=columns_to_remove, axis=1, inplace=True)
            
            # Calculate average revenue per room based on totalnetrevenue_room and total_rooms_sold
            data['avg_revenue_per_room'] = data['totalnetrevenue_room'] / data['total_rooms_sold']

            # Calculate the total guests per room based on totaladults and totalchildren
            data['total_guests_per_room'] = data['totaladults'] + data['totalchildren']
            
            # To understand the relationship between the trading_area feature and total_rooms_sold

            # Group data by 'trading_area' and calculate the mean of 'total_rooms_sold'
            rooms_sold_by_trading_area = data.groupby('trading_area')['total_rooms_sold'].mean()

            # Plot the average rooms sold in each trading area
            plt.figure(figsize=(15, 8))
            sns.barplot(x=rooms_sold_by_trading_area.index, y=rooms_sold_by_trading_area.values)
            plt.xlabel('Trading Area')
            plt.ylabel('Average Rooms Sold')
            plt.title('Average Rooms Sold by Trading Area')
            plt.xticks(rotation=75)
            plt.show()

            # Create a bar plot to visualize the relationship between 'air_conditioned_rooms' and 'total_rooms_sold'
            plt.figure(figsize=(15, 8))
            sns.barplot(x='air_conditioned_rooms', y='total_rooms_sold', data=data)
            plt.xlabel('Air Conditioned Rooms')
            plt.ylabel('Total Rooms Sold')
            plt.title('Air Conditioned Rooms vs. Total Rooms Sold')
            plt.show()

            # Create a bar plot to visualize the relationship between 'london_region_split' and 'total_rooms_sold'
            plt.figure(figsize=(15, 8))
            sns.barplot(x='london_region_split', y='total_rooms_sold', data=data)
            plt.xlabel('London Region Split')
            plt.ylabel('Total Rooms Sold')
            plt.title('London Region Split vs. Total Rooms Sold')
            plt.show()
            
            # Select numeric features for outlier detection
            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            # Create scatter plots for each feature against 'total_rooms_sold'
            plt.figure(figsize=(25, 30))
            for i, feature in enumerate(numeric_features):
                plt.subplot(6,5, i+1)
                sns.scatterplot(x='total_rooms_sold', y=feature, data=data)
                plt.title(f'{feature} vs. Total Rooms Sold')
                plt.xlabel('Total Rooms Sold')
                plt.ylabel(feature)
            plt.tight_layout()
            plt.show()

            # Create a correlation matrix for the selected numeric features
            correlation_matrix = data[numeric_features].corr()

            # Create a heatmap to visualize the correlation matrix
            plt.figure(figsize=(15, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()
            
            # Filter the correlation matrix to include only numerical features exclude the target feature
            numeric_features.remove('total_rooms_sold')

            correlation_matrix_numerical = data[numeric_features].corr()

            # Set the threshold for high correlation (0.8 or -0.8)
            threshold = 0.8

            # Create a list to store pairs of highly correlated features
            highly_correlated_features,features_variance = [],[]

            # Iterate through the correlation matrix and find highly correlated features
            for i in range(len(correlation_matrix_numerical.columns)):
                for j in range(i + 1, len(correlation_matrix_numerical.columns)):
                    if abs(correlation_matrix_numerical.iloc[i, j]) > threshold:
                        feature_i = correlation_matrix_numerical.columns[i]
                        feature_j = correlation_matrix_numerical.columns[j]
                        correlation_value = correlation_matrix_numerical.iloc[i, j]
                        highly_correlated_features.append((feature_i, feature_j, correlation_value))
                        
                        """
                        To decide which feature to keep, you can compare the variance of
                        each feature and choose the one with the higher variance:
                        """

                        var_feature_i = data[feature_i].var()
                        var_feature_j = data[feature_j].var()
                        features_variance.append((feature_i, feature_j, var_feature_i, var_feature_j))
                        
           
            # Display the highly correlated feature pairs and their correlation values
            print("multicollinearity:")
            for feature_i, feature_j, correlation_value in highly_correlated_features:
                print(f"{feature_i} and {feature_j} have a correlation of {correlation_value:.3f}")

            # Display the feature pairs and their variance values
            print("\n Variance:")
            for feature_i, feature_j, var_feature_i, var_feature_j in features_variance:
                print(f"\nVariance of {feature_i}", var_feature_i)
                print(f"Variance of {feature_j}", var_feature_j)
                
                try:
                    if var_feature_i > var_feature_j:
                        data = data.drop(feature_j, axis=1)
                    else:
                        data = data.drop(feature_i, axis=1)
                except KeyError as err:
                    pass

            print(f"\nSelected features", data.columns)
            
            # Create a correlation matrix for the selected numeric features

            # Select numeric features
            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            correlation_matrix = data[numeric_features].corr()

            # Create a heatmap to visualize the correlation matrix
            plt.figure(figsize=(15, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()


            # Identify relevant features from the datasets that could impact the final occupancy rate.

            # Calculate the correlation between each feature and the target variable "total_rooms_sold"
            correlation_matrix = data.corr()

            # Get the correlation values for the "total_rooms_sold" column with other features
            correlation_with_total_rooms_sold = correlation_matrix['total_rooms_sold']

            # Sort the correlation values in descending order
            correlation_with_total_rooms_sold = correlation_with_total_rooms_sold.sort_values(ascending=False)

            # Display the correlation values
            print(correlation_with_total_rooms_sold)
            
            
            # removing irrelevant columns with the reference of correlation matrix and scatter plot
            columns_to_remove = ['sellable_capacity','semi_flex_rate', 'mobile_web']
            data.drop(labels=columns_to_remove, axis=1, inplace=True)

            # Check and handle inconsistencies in data
            """
            Assumption: the following column should not have negative values
            """
            numberic_features_pos = list(data.select_dtypes(include=[np.number]).columns)
            numberic_features_neg = ['Canxrooms_last7days']
            numberic_features_both = ['rooms_remaining']

            numberic_features_pos = list(set(numberic_features_pos) - set(numberic_features_neg) - set(numberic_features_both))
            for feature in numberic_features_pos:
                if (data[feature] < 0).any():
                    print(f"Inconsistent data: {feature} has negative values.")

                    # Replace negative values by zero using a lambda function
                    data[feature] = data[feature].apply(lambda x: max(x, 0))
                
            for feature in numberic_features_neg:
                if (data[feature] > 0).any():
                    print(f"Inconsistent data: {feature} has positive values.")

                    # Replace negative values by zero using a lambda function
                    data[feature] = data[feature].apply(lambda x: min(x, 0))
                    
            data['family_rooms'] = data['family_rooms'].astype('int64')
            data['total_vws'] = data['total_vws'].astype('int64')
            data['total_rooms_sold'] = data['total_rooms_sold'].astype('int64')
            data['avgnights'] = data['avgnights'].astype('int64')
            data['totalchildren'] = data['totalchildren'].astype('int64')
            data['mobile_app'] = data['mobile_app'].astype('int64')
            data['corporate_booking_tool'] = data['corporate_booking_tool'].astype('int64')
            data['front_desk'] = data['front_desk'].astype('int64')
            data['travelport_gds'] = data['travelport_gds'].astype('int64')
            data['ccc'] = data['ccc'].astype('int64')
            data['agency'] = data['agency'].astype('int64')
            data['germany_web_de'] = data['germany_web_de'].astype('int64')
            data['amadeus_gds'] = data['amadeus_gds'].astype('int64')
            data['hub_mobile_app'] = data['hub_mobile_app'].astype('int64')
            data['booking_com'] = data['booking_com'].astype('int64')
            data['Canxrooms_last7days'] = data['Canxrooms_last7days'].astype('int64')
            data['off_rooms'] = data['off_rooms'].astype('int64')

            data['totalgrossrevenue_room'] = data['totalgrossrevenue_room'].astype('float64')
            data['totalnetrevenue_breakfast'] = data['totalnetrevenue_breakfast'].astype('float64')
            data['totalnetrevenue_mealdeal'] = data['totalnetrevenue_mealdeal'].astype('float64')
            data['flex_rate'] = data['flex_rate'].astype('float64')
            data['avg_revenue_per_room'] = data['avg_revenue_per_room'].astype('float64')
                        
            # Check summary statistics of each DataFrame
            print("\nSummary Statistics:")
            print(data.describe())
            
            # Outlier Detection in Data
            # Create subplots for box plots
            fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))

            # Flatten the axes array for easy iteration
            axes = axes.flatten()

            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            # Create box plots for each feature
            for i, feature in enumerate(numeric_features):
                sns.boxplot(x=feature,data=data, ax=axes[i])
                axes[i].set_title(f"{feature} vs Total Rooms Sold")
                axes[i].set_xlabel("Total Rooms Sold")
                axes[i].set_ylabel(feature)

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()

            
            # Handling Outliers using Z-Score 

            for feature in numeric_features:
                # Calculate the Z-Score for 'Total digital visits' column
                data['z_score'] = (data[feature] - data[feature].mean()) / data[feature].std()

                # Define the threshold for outliers (Z-Score of 3 or higher)
                threshold = 3

                # Replace outlier values with the mean value
                data[feature] = data[feature].where(data['z_score'].abs() < threshold, data[feature].mean())

                # Drop the Z-Score column if it's not needed anymore
                data.drop('z_score', axis=1, inplace=True)

            # Create subplots for box plots
            fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))

            # Flatten the axes array for easy iteration
            axes = axes.flatten()

            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            # Create box plots for each feature
            for i, feature in enumerate(numeric_features):
                sns.boxplot(x=feature,data=data, ax=axes[i])
                axes[i].set_title(f"{feature} vs Total Rooms Sold")
                axes[i].set_xlabel("Total Rooms Sold")
                axes[i].set_ylabel(feature)

            # Adjust layout and show the plot
            plt.tight_layout()
            plt.show()
            
            print("trading_area categories: ",data['trading_area'].unique())
            print("\nlondon_region_split categories: ",data['london_region_split'].unique())
            print("\nair_conditioned_rooms categories: ",data['air_conditioned_rooms'].unique())
            
            # Feature Encoding

            # Perform one-hot encoding 
            data = pd.get_dummies(data, columns=['air_conditioned_rooms','london_region_split'])

            # Initialize the LabelEncoder
            label_encoder = LabelEncoder()

            # Fit and transform the 'trading_area' column to numeric labels
            data['trading_area_encoded'] = label_encoder.fit_transform(data['trading_area'])
    
            # data for stay period
            print("First date for stay in data:", data['stay_date'].min())
            print("Last date for stay in data:", data['stay_date'].max())
            
            data = data.sort_values('stay_date')
            
            # Create lag features for time series forecasting
            def create_lag_features(data, lag_days=3):
                for i in range(1, lag_days + 1):
                    data[f'total_rooms_sold_lag_{i}'] = data['total_rooms_sold'].shift(i)
                return data

            data = create_lag_features(data)
            
            # Backward Fill
            data.fillna(method='bfill', inplace=True)
            
            # Create a correlation matrix for the selected numeric features

            # Select numeric features
            numeric_features = list(data.select_dtypes(include=[np.number]).columns)

            correlation_matrix = data[numeric_features].corr()

            # Create a heatmap to visualize the correlation matrix
            plt.figure(figsize=(15, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.show()

            
            # Removing the duplicate column
            data = data.drop(['trading_area','total_rooms_sold_lag_1', 'total_rooms_sold_lag_2',                              'air_conditioned_rooms_No','london_region_split_Regions'],axis=1)


            logging.info('Train Test Split Initiated')
            
            # Split the data into training and testing sets
            # Splitting the data into train and test set
            train = data[data['stay_date'].dt.day <= 24]
            test = data[data['stay_date'].dt.day > 24]

            # Removing the 'stay_date' column
            train = train.drop(['stay_date'],axis=1)
            test = test.drop(['stay_date'],axis=1)
            
            train.to_csv(self.data_transformation_config.train_data_path,index=False,header=True)
            test.to_csv(self.data_transformation_config.test_data_path,index=False,header=True)

            logging.info('Data Preparation is completed')
            
            logging.info(f'Train Dataframe Head : \n{train.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test.head().to_string()}')

            logging.info('Prepare independent featureset and target feature object')

            logging.info("Preprocessing is completed")
            
           
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=label_encoder
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train, test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise CustomException(e,sys)


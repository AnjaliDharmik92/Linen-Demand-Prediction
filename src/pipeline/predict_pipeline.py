import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/hotel-occupancy-prediction-model.pkl'
            model = load_object(file_path=model_path)
            pred = model.predict(features)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                total_vws:float,
                family_rooms:float,
                avgnights:float,
                totalchildren:float,
                totalgrossrevenue_room:float,
                totalnetrevenue_breakfast:float,
                totalnetrevenue_mealdeal:float,
                mobile_app:float,
                corporate_booking_tool:float,
                front_desk:float,
                ccc:float,
                travelport_gds:float,
                agency:float,
                germany_web_de:float,
                amadeus_gds :float
                hub_mobile_app:float,
                booking_com :float,
                Canxrooms_last7days:float,
                off_rooms:float,
                flex_rate:float,
                avg_revenue_per_room:float,
                air_conditioned_rooms_No:int,
                air_conditioned_rooms_Yes:int,
                london_region_split_Germany:int,
                london_region_split_London:int,
                london_region_split_Regions:int,
                trading_area_encoded:int,
                total_rooms_sold_lag_1:float,
                total_rooms_sold_lag_2:float,
                total_rooms_sold_lag_3:float):
                    
              
        
        self.total_vws = total_vws
        self.family_rooms = family_rooms
        self.avgnights = avgnights
        self.totalchildren = totalchildren
        self.totalgrossrevenue_room = totalgrossrevenue_room
        self.totalnetrevenue_breakfast = totalnetrevenue_breakfast
        self.totalnetrevenue_mealdeal = totalnetrevenue_mealdeal
        self.mobile_app = mobile_app
        self.corporate_booking_tool = corporate_booking_tool
        self.front_desk = front_desk
        self.ccc = ccc
        self.travelport_gds = travelport_gds
        self.agency = agency
        self.germany_web_de = germany_web_de 
        self.amadeus_gds = amadeus_gds
        self.hub_mobile_app = hub_mobile_app
        self.booking_com = booking_com
        self.Canxrooms_last7days = Canxrooms_last7days
        self.off_rooms = off_rooms
        self.flex_rate = flex_rate
        self.avg_revenue_per_room = avg_revenue_per_room
        self.air_conditioned_rooms_No = air_conditioned_rooms_No 
        self.air_conditioned_rooms_Yes = air_conditioned_rooms_Yes
        self.london_region_split_Germany = london_region_split_Germany
        self.london_region_split_London = london_region_split_London
        self.london_region_split_Regions = london_region_split_Regions
        self.trading_area_encoded = trading_area_encoded
        self.total_rooms_sold_lag_1 = total_rooms_sold_lag_1
        self.total_rooms_sold_lag_2 = total_rooms_sold_lag_2
        self.total_rooms_sold_lag_3 = total_rooms_sold_lag_3

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'total_vws':[self.total_vws], 
                'family_rooms':[self.family_rooms], 
                'avgnights':[self.avgnights], 
                'totalchildren':[self.totalchildren], 
                'totalgrossrevenue_room':[self.totalgrossrevenue_room], 
                'totalnetrevenue_breakfast':[self.totalnetrevenue_breakfast], 
                'totalnetrevenue_mealdeal':[self.totalnetrevenue_mealdeal], 
                'mobile_app':[self.mobile_app], 
                'corporate_booking_tool':[self.corporate_booking_tool], 
                'front_desk':[self.front_desk], 
                'ccc':[self.ccc], 
                'travelport_gds':[self.travelport_gds], 
                'agency':[self.agency], 
                'germany_web_de ':[self.germany_web_de], 
                'amadeus_gds':[self.amadeus_gds], 
                'hub_mobile_app':[self.hub_mobile_app], 
                'booking_com':[self.booking_com], 
                'Canxrooms_last7days':[self.Canxrooms_last7days], 
                'off_rooms':[self.off_rooms], 
                'flex_rate':[self.flex_rate], 
                'avg_revenue_per_room':[self.avg_revenue_per_room], 
                'air_conditioned_rooms_No':[self.air_conditioned_rooms_No], 
                'air_conditioned_rooms_Yes':[self.air_conditioned_rooms_Yes], 
                'london_region_split_Germany':[self.london_region_split_Germany], 
                'london_region_split_London':[self.london_region_split_London], 
                'london_region_split_Regions':[self.london_region_split_Regions], 
                'trading_area_encoded':[self.trading_area_encoded], 
                'total_rooms_sold_lag_1':[self.total_rooms_sold_lag_1], 
                'total_rooms_sold_lag_2':[self.total_rooms_sold_lag_2], 
                'total_rooms_sold_lag_3':[self.total_rooms_sold_lag_3]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
            
"""searching car for rent, also booking and canceling car rental, and changing time slot of rental"""
from datetime import date, datetime
from typing import Optional, Union, Type
import sqlite3

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field


class SearchCarRentalIntput(BaseModel):
    location: Optional[str] = Field(description="The location of the car rental.")
    name: Optional[str] = Field(description="The name of the car rental company.")
    price_tier: Optional[str] = Field(description="The price tier of the car rental.")
    start_date: Optional[Union[datetime, date]] = Field(description="The start date of the car rental.")
    end_date: Optional[Union[datetime, date]] = Field(description="The end date of the car rental.")


class SearchCarRentals(BaseTool):
    name: str = "search_car_rental"
    description: str = "Search for car rentals based on location, name, price tier, start date, and end date."
    args_schema: Type[BaseModel] = SearchCarRentalIntput
    return_direct: bool = False

    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self,
             location: Optional[str] = None,
             name: Optional[str] = None,
             price_tier: Optional[str] = None,
             start_date: Optional[Union[datetime, date]] = None,
             end_date: Optional[Union[datetime, date]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None,
             ) -> list[dict]:

        """
        Search for car rentals based on location, name, price tier, start date, and end date.

        Args:
            location (Optional[str]): The location of the car rental. Defaults to None.
            name (Optional[str]): The name of the car rental company. Defaults to None.
            price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
            start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

        Returns:
            list[dict]: A list of car rental dictionaries matching the search criteria.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        query = "SELECT * FROM car_rentals WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in results
        ]

    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()


class BookCarRentalInput(BaseModel):
    rental_id: int = Field(description="The ID of the car rental to book.")


class BookCarRental(BaseTool):
    name = "book_car_rental"
    description = "Search for car rentals based on location, name, price tier, start date, and end date."
    args_schema: Type[BaseModel] = BookCarRentalInput
    return_direct: bool = False
    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self, rental_id: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Book a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to book.

        Returns:
            str: A message indicating whether the car rental was successfully booked or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully booked."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()

class UpdateCarRentalInput(BaseModel):
    rental_id: int = Field(description="The ID of the car rental to update.")
    start_date: Optional[Union[datetime, date]] = Field(description="The new start date of the car rental.")
    end_date: Optional[Union[datetime, date]] = Field(description="The new end date of the car rental.")


class UpdateCarRental(BaseTool):
    name = "update_car_rental"
    description = "Update a car rental's start and end dates by its ID."
    args_schema: Type[BaseModel] = UpdateCarRentalInput
    return_direct: bool = False
    db: str = ""

    def __init__(self, db, ):
        super().__init__()
        self.db = db

    def _run(self,
             rental_id: int,
             start_date: Optional[Union[datetime, date]] = None,
             end_date: Optional[Union[datetime, date]] = None,
             run_manager: Optional[CallbackManagerForToolRun] = None,
             ) -> str:
        """
        Update a car rental's start and end dates by its ID.

        Args:
            rental_id (int): The ID of the car rental to update.
            start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

        Returns:
            str: A message indicating whether the car rental was successfully updated or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date, rental_id),
            )
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
            )

        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully updated."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()

class CancelCarRentalInput(BaseModel):
    rental_id: int = Field(description="The ID of the car rental to cancel.")


class CancelCarRental(BaseTool):
    name = "cancel_car_rental"
    description = "Cancel a car rental by its ID."
    args_schema: Type[BaseModel] = CancelCarRentalInput
    return_direct: bool = False
    db: str = ""

    def __init__(self, db):
        super().__init__()
        self.db = db

    def _run(self, rental_id: int) -> str:
        """
        Cancel a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to cancel.

        Returns:
            str: A message indicating whether the car rental was successfully cancelled or not.
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        conn.commit()

        if cursor.rowcount > 0:
            conn.close()
            return f"Car rental {rental_id} successfully cancelled."
        else:
            conn.close()
            return f"No car rental found with ID {rental_id}."
    @staticmethod
    def get_file_docstring():
        return get_tool_group_desc()

def get_tool_group_desc():
    return __doc__

if __name__ == "__main__":
    print(__doc__)
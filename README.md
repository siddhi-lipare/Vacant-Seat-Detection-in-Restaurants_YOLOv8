# Vacant_Seat_Detection_YOLOv8
Developing a Computer Vision based System which captures live footage from the CCTV Camera in campus restaurants to detect empty seats.

*Motivation*
----------
As the intake in IISER - B is increasing day by day, the crowd in the campus restaurants such as Sudarshan Restaurant, Indian Coffee House (ICH) is increasing as well. A significant number of students, often commuting from distant hostel premises, encounter challenges in securing seating arrangements at Sudarshan restaurant due to high occupancy. Some students who visit the mess, try some food to realise that the food is too spicy and are left with no option but to eat at these restaurants instead. Furthermore, during exam days when maximum students study in the Library, they prefer having lunch from Sudarshan or ICH to save their time. If there is no space, they have to either go back or wait till they find a seat. This consumes a lot of time in their already busy schedules. To solve this issue, it would be helpful to know ahead of time if there are empty tables or seats available.

----------
![ouutput](https://github.com/siddhi-lipare/Vacant-Seat-Detection-in-Restaurants/assets/98029886/035f5dcd-133a-47eb-b4ae-d7ebe83fbc0b)



----------
*Problem Statement*
----------
- Our challenge is to build a Computer Vision based System which captures live footage from the CCTV Camera in Sudarshan Restaurant and detect empty tables. 
- Empty tables will be surrounded with a green bounding box in the video and Occupied ones as Red.
- This system can later be used for ICH, Library for the same problem statement after we've dealt with the challenges faced during Step 1: Sudarshan Restaurant.
- Our plan is to display the live status on a website/app for all the IISERB community to access.

----------
*Prerequisites*
-------------
- `Python 3.10` 
- Install necessary packages using `pip install -r requirements.txt`
----------
- Run the Following command:
```bash
python main_image.py
```
and
```bash
python main_video.py
```
View `output_image.jpg` and `outputvid.mp4`

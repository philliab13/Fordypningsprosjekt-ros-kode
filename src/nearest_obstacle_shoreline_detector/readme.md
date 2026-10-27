So I will be taking all the lidar points, filtering out the points that are above 2 meter. 
Then flattening the points down to the waterplane. Then I have all the points in the waterplane.
Then I would need to create a polyline between the points, I dont need all the points far inland, take inspiration from the stixel based free space estimation. 
Then i have a nearest obstacle shoreline detector which I can use. 
Boats will be present, and I have footprint detector running in parallell, which makes it good for seeing boats too. So the only thing this needs to do is to create a barrier for the shore. 
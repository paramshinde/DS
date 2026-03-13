import pandas as pd

data = {
"bedroom":[3,4,2,3,5,4,3,2,4,3],
"bathroom":[1,2,1,2,3,2,1,1,3,2],
"sqft_living":[1180,2100,950,1600,3000,2400,1400,800,2800,1750],
"floors":[2,2,1,2,2,2,1,1,2,2],
"grade":[6,7,5,7,9,8,6,5,8,7],
"sqft_above":[1180,2100,950,1600,3000,2400,1400,800,2800,1750],
"sqft_base":[400,500,300,450,800,600,400,250,700,450],
"price":[221900,350000,180000,275000,520000,410000,230000,150000,470000,300000]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("house_data.csv", index=False)

print(df)
"https://drive.google.com/drive/mobile/folders/1uIVSglL-xTCZDKztrwjtgEzQL9yrF2QH/1AQU4sWyhSMORHVrOSLzGfXraADSsD2He/1yj9po9uQDpwOvuTT8MsydX3pN71hM2vk/1SzgMU-ILQtZoqFTJNnQ8qZdJ0AcP8sDZ?usp=sharing&sort=13&direction=a"
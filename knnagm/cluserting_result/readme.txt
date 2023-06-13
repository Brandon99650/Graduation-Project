兩個資料夾:
K_5: torch_geomteric.data.Data, 用code 裡面 getdata.getdata的函數就可以讀出 (neighbor: nearest 5 )

K_5_7: K_5 這個graph分7群:內部:
	clustering_result:
		eachC:	每一群有哪些人
		kw:	每一群的人有哪些keywords
		dupkw:	每一群出現率最高(top15) 的keyword

		cluster.csv: 總檔案，紀錄所有人每個人都在哪群

	cluser.pt: 分群的label，0代表他不屬於該群，1代表它屬於該群 (是一個torch 的tensor, 使用torch.load 讀取)
	
	K_5_7.pt: trained gcn model
	

		
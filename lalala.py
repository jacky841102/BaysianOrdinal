def haha(Sigma):
    for row in Sigma:
        str = "ask1 rvwrid_%d" % row
        for d in range(50):
            try:
                str += (" assgnid_%d >" % Sigma[row][d])
            except:
                pass
        print(str)

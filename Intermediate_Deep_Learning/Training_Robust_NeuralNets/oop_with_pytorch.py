class BankAccount:
    __balance = 0
    def __init__(self, balance):
        self.__balance = balance

    def deposit(self,amount):
        self.__balance += amount

    def showBalance(self):
        print("Balance: ",self.__balance)

account = BankAccount(1000)
account.deposit(100)
print(account.showBalance())
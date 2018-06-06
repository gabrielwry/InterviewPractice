class DoYouKnowRecursion:
    def __init__(self,recursion_learned):
        self.recursion_learned = True

    def learn_recursion(self):
        try:
            assert self.recursion_learned == False
            print 'You are an idiot'
            return self.learn_recursion()
        except AssertionError:
            print "Don't be a fool, you don't know recursion!"
            self.recursion_learned = False
            return self.learn_recursion()
        except RuntimeError:
            print 'Give it up, you idiot.'


if __name__ == "__main__":
    what_i_did_this_afternoon = DoYouKnowRecursion(False)
    what_i_did_this_afternoon.learn_recursion()
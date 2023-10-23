# Custom exceptions for the Eyring model

class DistributionError(Exception):
    '''Exception raised for errors when generating a distribution
    
    Attributes:
        message -- error message

    '''

    def __init__(self, message='Error in generating a distribution'):
        self.message = message
        super().__init__(self.message)


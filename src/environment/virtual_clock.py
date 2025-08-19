class VirtualClock:
    """虛擬時鐘類，用於模擬環境中的時間管理"""
    
    def __init__(self, start_time=0):
        """
        初始化虛擬時鐘
        
        Args:
            start_time: 開始時間，默認為0
        """
        self._current_time = start_time
    
    def time(self):
        """
        獲取當前時間
        
        Returns:
            int: 當前時間
        """
        return self._current_time
    
    def advance(self, duration):
        """
        推進時間
        
        Args:
            duration: 要推進的時間長度
            
        Returns:
            int: 推進後的當前時間
        """
        self._current_time += duration
        return self._current_time
    
    def reset(self, start_time=0):
        """
        重置時鐘
        
        Args:
            start_time: 重置後的開始時間，默認為0
        """
        self._current_time = start_time
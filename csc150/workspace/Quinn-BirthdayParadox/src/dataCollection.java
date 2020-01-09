
public class dataCollection {

	/**
	 * This goes through and checks for duplicate birthdays in each room
	 * then returns if there are or not.
	 */
	
    public byte birthSort(int[] birthDates) {
        for (int i = 0; i < birthDates.length; i++) {
            for (int j = i + 1; j < birthDates.length - i; j++) {
                if (birthDates[i] == birthDates[j]) {
                    return 1; 
                }
            }
        }
        return 0;
    }

    /**
     * This makes the birthdays for the people put into the rooms
     * and then returns an array of them
     */
    
    public int[] birthGeneration(int[] birthDates) {
        for (byte j = 0; j < birthDates.length; j++) {
            birthDates[j] = (int)(366 * Math.random());
        }
        return birthDates;
    }
    
    /**
     * This makes an array that acts as a room 
     * and then returns it
     */
    
    public int[] arrayBuilder(int roomSize) {
        int[] birthArray = new int[roomSize]; 
        return birthArray;
    }
}

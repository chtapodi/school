public class BirthdayParadox {

    /**
     * Xavier Quinn
     * 
     * I affirm that I have carried out the attached academic endeavors with full academic honesty, in
     * accordance with the Union College Honor Code and the course syllabus.
     */

    public static void main(String[] args) {
    	dataCollection tester = new dataCollection();
    	
    	/**
    	 * This part creates rooms and uses the methods in the dataCollection class to run experiments on the rooms.
    	 * After it cycles through all the rooms it returns the values.
    	 */
    	
        for (int roomSize=5;roomSize <= 100; roomSize=roomSize+5) {
        	int[] birthArray = tester.arrayBuilder(roomSize);
        	int score = 0;

            for (int i = 0; i < 10; i++) {
                score = score + tester.birthSort(tester.birthGeneration(birthArray));
            }
            System.out.println("The test worked " + score + "/10 for " + roomSize + " people");
        }
    }
}

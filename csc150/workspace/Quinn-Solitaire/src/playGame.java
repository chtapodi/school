public class playGame {

	/**
	 * This tests out the rules of the solitaire game 
	 * @author xavier
	 *
	 *I affirm that I have carried out the attached 
	 *academic endeavors with full academic honesty, in
	 *accordance with the Union College Honor Code and
	 *the course syllabus.
	 *
	 */
	public static void main(String[] args) {
		simulator game = new simulator();
		double score=0.0;
		int tests=1000;
		game.playgame();
		
		while(tests<=10000) {
			for(int j=0;j<=tests;j++) {
				if(game.playgame()) {
					score++;
				}
			}
			
			System.out.println((int)(score) + "/" + tests + " = " + (int)((score/tests)*100) + "%");
			tests=tests+1000;
			score=0;
		}
	}

}

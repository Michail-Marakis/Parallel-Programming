library ieee;
use ieee.std_logic_1164.all;

ENTITY Full_adder IS
    PORT(A, B, Cin : IN STD_LOGIC;
         Sum, Cout : OUT STD_LOGIC);

END Full_adder;

ARCHITECTURE Full_adder_arch OF Full_adder IS
    
	 SIGNAL S1, S2, S3, S4 : STD_LOGIC;

BEGIN
    
	 S1 <= A AND (NOT B) AND (NOT Cin);
    S2 <= (NOT A) AND B AND (NOT Cin);				--create full adder
    S3 <= (NOT A) AND (NOT B) AND Cin;
    S4 <= A AND B AND Cin;
    Sum <= S1 OR S2 OR S3 OR S4;
    Cout <= (B AND Cin) OR (A AND Cin) OR (A AND B);

END Full_adder_arch;


library ieee;
use ieee.std_logic_1164.all;


ENTITY Part1 IS
    PORT ( A, B, AINVERT, BINVERT, Cin : IN STD_LOGIC;
           OPERATION : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
           Cout, F : OUT STD_LOGIC);
END Part1;

ARCHITECTURE Part1_arch OF Part1 IS
    COMPONENT Full_adder IS
        PORT (A, B, Cin : IN STD_LOGIC;
              Sum, Cout : OUT STD_LOGIC);
    END COMPONENT;

    SIGNAL AI, BI, S : STD_LOGIC;

BEGIN
    AI <= A WHEN AINVERT = '0' ELSE NOT A;
    BI <= B WHEN BINVERT = '0' ELSE NOT B;					--circuit starts with multiplexers 2-1 for a,b or the inverts

    A3: Full_adder PORT MAP(AI, BI, Cin, S, Cout);

    WITH OPERATION SELECT
        F <= AI AND BI WHEN "00",
             AI OR BI WHEN "01",										-- choice for result of multiplexer 4-1 
             S WHEN "10",
             AI XOR BI WHEN "11";

END Part1_arch;

	

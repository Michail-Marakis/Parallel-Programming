library ieee;
use ieee.std_logic_1164.all;

ENTITY ALU16bit IS
    PORT ( A, B : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
           Ainvert, Binvert, CIN : IN STD_LOGIC;
           OPERATION : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
           result : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
           COUT, overflow : BUFFER STD_LOGIC
    );
END ALU16bit;

ARCHITECTURE ALU16bit_arch OF ALU16bit IS
    COMPONENT Part1 IS   													--dimiourgoume component apo to part1, alu-1bit gia na thn xrisimopoihsoyme me tis prajeis mas bit me bit
        PORT ( A, B, AINVERT, BINVERT, Cin : IN STD_LOGIC;
               OPERATION : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
               Cout, F : OUT STD_LOGIC
        );
    END COMPONENT;

    SIGNAL C : STD_LOGIC_VECTOR(15 DOWNTO 0);
																--ta simata apo slice se slice
	 SIGNAL S : STD_LOGIC_VECTOR(15 DOWNTO 0);

BEGIN
    -- ylopoihsh prwtou slice
    first_slice: Part1 PORT MAP (A(0), B(0), Ainvert, Binvert, CIN, OPERATION, C(0), S(0));

    -- ftiaxnoume kai ta ypoloipa slice
    other_slices: FOR i IN 1 TO 15 GENERATE
    slices_ALU : Part1 PORT MAP (A(i), B(i), Ainvert, Binvert, C(i-1), OPERATION, C(i), S(i));
    
	 END GENERATE;

    result <= S;
    
	 COUT <= C(15);
    
	 overflow <= C(14) XOR COUT;

END ALU16bit_arch;


library ieee;
use ieee.std_logic_1164.all;

ENTITY prajeis IS
    PORT ( opcode : IN STD_LOGIC_VECTOR(2 DOWNTO 0);	--to opcode exei mikos 3
           Ainvert, Binvert, CARRYIN : OUT STD_LOGIC;
           OPERATION : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)  --to operation exei mikos 2
			  ); 
END prajeis;

ARCHITECTURE prajeis_arch OF prajeis IS
BEGIN
    PROCESS(opcode)			--ylopoihsh twn prajewn
    BEGIN
        CASE opcode IS
            WHEN "000" => -- and
                Ainvert <= '0';
                Binvert <= '0';
                CARRYIN <= '0';
                OPERATION <= "00";
            WHEN "001" => -- or
                Ainvert <= '0';
                Binvert <= '0';
                CARRYIN <= '0';
                OPERATION <= "01";
            WHEN "010" => -- arithmitiki prosthesh
                Ainvert <= '0';
                Binvert <= '0';
                CARRYIN <= '0';
                OPERATION <= "10";
            WHEN "011" => -- arithmitiki afairesh
                Ainvert <= '0';
                Binvert <= '1';
                CARRYIN <= '1';
                OPERATION <= "10";
            WHEN "100" => -- nor
                Ainvert <= '1';
                Binvert <= '1';
                CARRYIN <= '0';
                OPERATION <= "00";
            WHEN "101" => -- nand
                Ainvert <= '1';
                Binvert <= '1';
                CARRYIN <= '0';
                OPERATION <= "01";
            WHEN "110" => -- xor
                Ainvert <= '0';
                Binvert <= '0';
                CARRYIN <= '0';
                OPERATION <= "11";
            WHEN OTHERS => --den xrisimopoihte to opcode = 111
                Ainvert <= '0';
                Binvert <= '0';
                CARRYIN <= '0';
                OPERATION <= "00";
        END CASE;
    
	 END PROCESS;

END prajeis_arch;

library ieee;
use ieee.std_logic_1164.all;

ENTITY Part_2 IS
    PORT ( A, B : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
           opcode : IN STD_LOGIC_VECTOR(2 DOWNTO 0); 			--orismos koryfaias entity
           result : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
           COUT, overflow : OUT STD_LOGIC
    );
END Part_2;

ARCHITECTURE Part_2_arch OF Part_2 IS
    
	 SIGNAL internal_COUT : STD_LOGIC; 

    COMPONENT ALU16bit IS
        PORT ( A, B : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
               Ainvert, Binvert, CIN : IN STD_LOGIC;
               OPERATION : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
               result : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
               COUT, overflow : OUT STD_LOGIC
        );
    END COMPONENT;															--orizoune ws components ta entities poy ftiajame

    COMPONENT prajeis IS
        PORT ( opcode : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
               Ainvert, Binvert, CARRYIN : OUT STD_LOGIC;
               OPERATION : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
        );
    END COMPONENT;

    SIGNAL Ainvert, Binvert, CARRYIN : STD_LOGIC;
    
	 SIGNAL OPERATION : STD_LOGIC_VECTOR(1 DOWNTO 0);

BEGIN
    stage1: prajeis PORT MAP (opcode, Ainvert, Binvert, CARRYIN, OPERATION);
																																						--
	 stage2: ALU16bit PORT MAP (A, B, Ainvert, Binvert, CARRYIN, OPERATION, result, internal_COUT, overflow);

    COUT <= internal_COUT; 														--apo edw tha prokypsei teleftaia timh kratoumenoy gia elegxo overflow

END Part_2_arch;
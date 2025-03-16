/**
 * Enum que representa los diferentes tipos de ataques de Phantom Attacker.
 */
export enum PhantomAttackerAttack {
    DOS = "dos",
    DDOS = "ddos",
    SCANNING = "scanning",
}

export namespace PhantomAttackerAttack {
    /**
     * Lista de ataques de Phantom Attacker.
     */
    export const Types: ReadonlyArray<PhantomAttackerAttack> = [
        PhantomAttackerAttack.DOS,
        PhantomAttackerAttack.DDOS,
        PhantomAttackerAttack.SCANNING,
    ];

    /**
     * Convierte un string a un tipo de ataque de Phantom Attacker.
     *
     * @param type String que representa un tipo de ataque de Phantom Attacker.
     * @returns Tipo de ataque de Phantom Attacker.
     */
    export const toString = (type: PhantomAttackerAttack): string => {
        switch (type) {
            case PhantomAttackerAttack.DOS:
                return "DoS";
            case PhantomAttackerAttack.DDOS:
                return "DDoS";
            case PhantomAttackerAttack.SCANNING:
                return "Scanning";
        }
    };

    /**
     * Convierte un tipo de ataque de Phantom Attacker a un icono.
     *
     * @param type Tipo de ataque de Phantom Attacker.
     * @returns Icono.
     */
    export const toIcon = (type: PhantomAttackerAttack): string => {
        switch (type) {
            case PhantomAttackerAttack.DOS:
                return "lucideBomb";
            case PhantomAttackerAttack.DDOS:
                return "lucideCrosshair";
            case PhantomAttackerAttack.SCANNING:
                return "lucideRadar";
        }
    };
}

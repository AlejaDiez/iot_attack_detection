/**
 * Enum que representa los diferentes tipos de modelos de Cyber Shield.
 */
export enum CyberShieldModel {
    TENSORFLOW = "tensorflow",
    FEDERATED_LEARNING = "federated-learning",
}

export namespace CyberShieldModel {
    /**
     * Lista de modelos de Cyber Shield.
     */
    export const Types: ReadonlyArray<CyberShieldModel> = [
        CyberShieldModel.TENSORFLOW,
        CyberShieldModel.FEDERATED_LEARNING,
    ];

    /**
     * Convierte un string a un tipo de modelo de Cyber Shield.
     *
     * @param type String que representa un modelo de Cyber Shield.
     * @returns Modelo de Cyber Shield.
     */
    export const toString = (type: CyberShieldModel): string => {
        switch (type) {
            case CyberShieldModel.TENSORFLOW:
                return "TensorFlow";
            case CyberShieldModel.FEDERATED_LEARNING:
                return "Federaed Learning";
        }
    };
}

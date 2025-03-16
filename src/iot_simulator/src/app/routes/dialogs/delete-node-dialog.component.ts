import { CommonModule } from "@angular/common";
import { Component } from "@angular/core";
import { ReactiveFormsModule } from "@angular/forms";
import { Node } from "@models/node";
import { BrnDialogRef, injectBrnDialogContext } from "@spartan-ng/brain/dialog";
import { BrnSelectImports } from "@spartan-ng/brain/select";
import { HlmButtonDirective } from "@spartan-ng/ui-button-helm";
import {
    HlmDialogDescriptionDirective,
    HlmDialogFooterComponent,
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,
} from "@spartan-ng/ui-dialog-helm";
import { HlmSelectModule } from "@spartan-ng/ui-select-helm";

export interface DeleteNodeDialogContext {
    node: Node;
}

@Component({
    selector: "app-delete-node-dialog",
    imports: [
        BrnSelectImports,
        CommonModule,
        ReactiveFormsModule,
        HlmButtonDirective,
        HlmDialogDescriptionDirective,
        HlmDialogHeaderComponent,
        HlmDialogTitleDirective,
        HlmDialogFooterComponent,
        HlmSelectModule,
    ],
    template: `
        <hlm-dialog-header class="w-96">
            <h3 hlmDialogTitle>¿Estás seguro de eliminar {{ name }}?</h3>
        </hlm-dialog-header>
        <p hlmDialogDescription>
            Si elimina este nodo, se eliminará de forma permanente y no podrá
            recuperarlo.
        </p>
        <hlm-dialog-footer>
            <button hlmBtn variant="outline" (click)="cancel()">
                Cancelar
            </button>
            <button hlmBtn variant="destructive" (click)="delete()">
                Eliminar
            </button>
        </hlm-dialog-footer>
    `,
    host: { class: "flex flex-col gap-4" },
})
export class DeleteNodeDialogComponent {
    private readonly _context!: DeleteNodeDialogContext;
    protected get name(): string {
        return this._context.node.name;
    }

    public constructor(
        private readonly _ref: BrnDialogRef<DeleteNodeDialogContext>,
    ) {
        this._context = injectBrnDialogContext<DeleteNodeDialogContext>();
    }

    protected cancel() {
        this._ref.close();
    }

    protected delete() {
        this._ref.close({ node: this._context.node });
    }
}

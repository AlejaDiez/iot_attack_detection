import { CommonModule } from "@angular/common";
import { Component, computed, inject, Signal } from "@angular/core";
import { ReactiveFormsModule } from "@angular/forms";
import { Node } from "@models/node";
import { TranslateModule } from "@ngx-translate/core";
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
        TranslateModule,
    ],
    template: `
        <hlm-dialog-header class="w-96">
            <h3 hlmDialogTitle>
                {{ "DELETE_NODE" | translate: { name: name() } }}
            </h3>
        </hlm-dialog-header>
        <p hlmDialogDescription>
            {{ "DELETE_DESC" | translate }}
        </p>
        <hlm-dialog-footer>
            <button hlmBtn variant="outline" (click)="cancel()">
                {{ "CANCEL" | translate }}
            </button>
            <button hlmBtn variant="destructive" (click)="delete()">
                {{ "DELETE" | translate }}
            </button>
        </hlm-dialog-footer>
    `,
    host: { class: "flex flex-col gap-4" },
})
export class DeleteNodeDialogComponent {
    protected readonly ref: BrnDialogRef<DeleteNodeDialogContext> = inject(
        BrnDialogRef<DeleteNodeDialogContext>,
    );
    protected readonly context =
        injectBrnDialogContext<DeleteNodeDialogContext>();
    protected readonly name: Signal<string> = computed(
        () => this.context.node.name,
    );

    protected cancel() {
        this.ref.close();
    }

    protected delete() {
        this.ref.close({ node: this.context.node });
    }
}

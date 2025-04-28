import {
    AfterViewInit,
    ChangeDetectorRef,
    Component,
    input,
    InputSignal,
} from "@angular/core";
import { RouterModule, RouterOutlet } from "@angular/router";
import { HlmDialogCloseDirective } from "@components/ui/ui-dialog-helm/src";
import { Node, NodeType } from "@models/node";
import { NgIcon, provideIcons } from "@ng-icons/core";
import { lucideGhost, lucideX } from "@ng-icons/lucide";
import { HlmButtonModule } from "@spartan-ng/ui-button-helm";
import { HlmCardDirective, HlmCardModule } from "@spartan-ng/ui-card-helm";
import { fadeAnimation } from "../../app.routes.transition";

@Component({
    selector: "app-panel-node",
    imports: [
        HlmButtonModule,
        HlmCardModule,
        HlmDialogCloseDirective,
        NgIcon,
        RouterModule,
    ],
    providers: [provideIcons({ lucideGhost, lucideX })],
    templateUrl: "panel-node.component.html",
    host: {
        class: "fixed right-5 top-1/2 flex flex-col max-h-[calc(100%-2.5rem)] min-w-[512px] max-w-[calc(100%-2.5rem)] -translate-y-1/2 rounded-xl overflow-hidden",
    },
    hostDirectives: [HlmCardDirective],
    animations: [fadeAnimation],
})
export class PanelNodeComponent implements AfterViewInit {
    protected readonly node: InputSignal<Node> = input.required<Node>();
    protected readonly NodeType: typeof NodeType = NodeType;
    private _loaded: boolean = false;

    constructor(private readonly cdr: ChangeDetectorRef) {}

    public ngAfterViewInit(): void {
        this._loaded = true;
        this.cdr.detectChanges();
    }

    protected getAnimationData(outlet: RouterOutlet) {
        if ((outlet && !outlet.isActivated) || !this._loaded) return "_";
        return outlet.activatedRoute.snapshot.url[0]?.path ?? "_";
    }
}
